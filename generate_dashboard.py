#!/usr/bin/env python3
"""Rebuild the Claude Code Pilot Register dashboard from live AWS data.

Reads Bedrock model-invocation logs from the s3://boston-claude-pilot-logs
bucket, attributes every call to a person via the invoking IAM user, and maps
that person to a team from the IAM user's `Team` tag.  The aggregated result is
injected into dashboard_template.html and written out as dashboard.html.

Usage:  python3 generate_dashboard.py [-o dashboard.html]

Requires boto3 and AWS credentials with s3:GetObject on the log bucket plus
iam:ListUsers / iam:ListUserTags.
"""

import argparse
import collections
import datetime as dt
import gzip
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
from botocore.config import Config

BUCKET = "boston-claude-pilot-logs"
LOG_PREFIX = "AWSLogs/"
TZ = ZoneInfo("America/New_York")
SESSION_GAP = 30 * 60  # seconds of inactivity that start a new session
WORKERS = 48


# --------------------------------------------------------------------------
# pricing
# --------------------------------------------------------------------------

def rates(model):
    """USD per million tokens: (input, output, cache_read, cache_write).

    Models absent from this table are billed at zero -- non-Anthropic models on
    the account sit outside the pilot's cost envelope.
    """
    m = model.lower()
    if "opus" in m:      return (5.0, 25.0, 0.50, 6.25)
    if "sonnet-5" in m:  return (2.0, 10.0, 0.20, 2.50)
    if "sonnet" in m:    return (3.0, 15.0, 0.30, 3.75)
    if "haiku-4-5" in m: return (1.0, 5.0, 0.10, 1.25)
    if "haiku" in m:     return (0.8, 4.0, 0.08, 1.00)
    return (0.0, 0.0, 0.0, 0.0)


def family(model):
    m = model.lower()
    for f in ("opus", "sonnet", "haiku"):
        if f in m:
            return f
    return "other"


def normalize_model(model_id):
    """`arn:...:inference-profile/us.anthropic.claude-opus-5:0` -> `us.anthropic.claude-opus-5`."""
    name = (model_id or "").split("/")[-1]
    return name[:-2] if name.endswith(":0") else name


# --------------------------------------------------------------------------
# IAM: who is on which team
# --------------------------------------------------------------------------

def load_teams(iam):
    """Map IAM user name -> Team tag value, skipping users with no Team tag."""
    teams = {}
    for page in iam.get_paginator("list_users").paginate():
        for user in page["Users"]:
            name = user["UserName"]
            tags = {t["Key"]: t["Value"] for t in iam.list_user_tags(UserName=name)["Tags"]}
            if tags.get("Team"):
                teams[name] = tags["Team"]
    return teams


# --------------------------------------------------------------------------
# S3: read the invocation logs
# --------------------------------------------------------------------------

def log_keys(s3):
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=LOG_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # `data/` holds oversized request bodies offloaded from the log
            # lines; every field we need is inline in the log line itself.
            if "/data/" not in key and key.endswith(".json.gz"):
                keys.append(key)
    return keys


def parse_line(line):
    rec = json.loads(line)
    inp = rec.get("input") or {}
    out = rec.get("output") or {}
    body_in = inp.get("inputBodyJson")
    body_in = body_in if isinstance(body_in, dict) else {}
    body_out = out.get("outputBodyJson")

    tools = []
    if isinstance(body_out, list):
        # Streamed responses arrive as a list of SSE chunks.
        for chunk in body_out:
            if isinstance(chunk, dict) and chunk.get("type") == "content_block_start":
                block = chunk.get("content_block") or {}
                if block.get("type") == "tool_use" and block.get("name"):
                    tools.append(block["name"])
    elif isinstance(body_out, dict):
        for block in body_out.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name"):
                tools.append(block["name"])

    # Claude Code pings Bedrock with a one-token "." prompt to validate
    # credentials; those are not real usage.
    first_text = None
    messages = body_in.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        content = messages[0].get("content")
        if isinstance(content, str):
            first_text = content
        elif isinstance(content, list) and content and isinstance(content[0], dict):
            first_text = content[0].get("text")

    return {
        "ts": rec.get("timestamp"),
        "arn": (rec.get("identity") or {}).get("arn"),
        "model": rec.get("modelId"),
        "err": rec.get("errorCode"),
        "it": inp.get("inputTokenCount") or 0,
        "ot": out.get("outputTokenCount") or 0,
        "rt": inp.get("cacheReadInputTokenCount") or 0,
        "wt": inp.get("cacheWriteInputTokenCount") or 0,
        "ping": first_text == ".",
        "tools": tools,
    }


def fetch_invocations(session, cfg, keys, progress=True):
    local = threading.local()
    lock = threading.Lock()
    invocations = []
    done = [0]

    def client():
        if not hasattr(local, "c"):
            local.c = session.client("s3", config=cfg)
        return local.c

    def handle(key):
        blob = client().get_object(Bucket=BUCKET, Key=key)["Body"].read()
        try:
            text = gzip.decompress(blob).decode("utf-8", "replace")
        except OSError:
            text = blob.decode("utf-8", "replace")
        parsed = []
        for line in text.strip().split("\n"):
            if line.strip():
                try:
                    parsed.append(parse_line(line))
                except (ValueError, TypeError):
                    continue
        with lock:
            invocations.extend(parsed)
            done[0] += 1
            if progress and done[0] % 2000 == 0:
                print(f"  ...{done[0]}/{len(keys)} log files", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(handle, keys))
    return invocations


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def blank():
    return dict(calls=0, cost=0.0, input_tokens=0, output_tokens=0,
                cache_read_tokens=0, cache_write_tokens=0, input_cost=0.0,
                output_cost=0.0, cache_read_cost=0.0, cache_write_cost=0.0)


def build_data(invocations, teams, account_id, generated_at):
    cutoff = generated_at.strftime("%Y-%m-%dT%H:%M:%S")
    rows = [r for r in invocations if r["ts"] and r["ts"] < cutoff]
    raw_count = len(rows)

    # Exclusions, in order: transport/service errors, credential pings, then
    # anyone whose IAM user carries no Team tag (service accounts, test users).
    errored = sum(1 for r in rows if r["err"])
    rows = [r for r in rows if not r["err"]]
    pings = sum(1 for r in rows if r["ping"])
    rows = [r for r in rows if not r["ping"]]
    untagged = sum(1 for r in rows if (r["arn"] or "").split("/")[-1] not in teams)
    rows = [r for r in rows if (r["arn"] or "").split("/")[-1] in teams]

    records, raw_costs, raw_tools = [], [], []
    for r in rows:
        user = r["arn"].split("/")[-1]
        model = normalize_model(r["model"])
        p_in, p_out, p_read, p_write = rates(model)
        it, ot, rt, wt = r["it"], r["ot"], r["rt"], r["wt"]
        utc = dt.datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        local_t = utc.astimezone(TZ)
        cost = (it / 1e6 * p_in, ot / 1e6 * p_out, rt / 1e6 * p_read, wt / 1e6 * p_write)
        rec = {
            "d": local_t.strftime("%Y-%m-%d"), "w": local_t.weekday(), "h": local_t.hour,
            "ts": r["ts"], "u": user, "t": teams[user], "m": family(model),
            "ic": round(cost[0], 6), "oc": round(cost[1], 6),
            "rc": round(cost[2], 6), "wc": round(cost[3], 6),
            "it": it, "ot": ot, "rt": rt, "wt": wt,
        }
        if r["tools"]:
            rec["tl"] = sorted(set(r["tools"]))
        records.append((utc.timestamp(), model, rec, cost, r["tools"]))

    records.sort(key=lambda x: (x[0], x[2]["u"], x[1], x[2]["it"], x[2]["ot"],
                                x[2]["rt"], x[2]["wt"]))

    by_day, by_user, by_team, by_model = (collections.defaultdict(blank) for _ in range(4))
    tool_calls, tool_users = collections.Counter(), collections.defaultdict(set)
    user_days, user_team, team_users = collections.defaultdict(set), {}, collections.defaultdict(set)
    heatmap = [[0] * 24 for _ in range(7)]
    sessions, last_seen = collections.Counter(), {}

    def add(acc, rec, cost):
        ic, oc, rc, wc = cost
        acc["calls"] += 1
        acc["input_tokens"] += rec["it"]; acc["output_tokens"] += rec["ot"]
        acc["cache_read_tokens"] += rec["rt"]; acc["cache_write_tokens"] += rec["wt"]
        acc["input_cost"] += ic; acc["output_cost"] += oc
        acc["cache_read_cost"] += rc; acc["cache_write_cost"] += wc
        acc["cost"] += ic + oc + rc + wc

    for epoch, model, rec, cost, tools in records:
        user, team, day = rec["u"], rec["t"], rec["d"]
        add(by_day[day], rec, cost); add(by_user[user], rec, cost)
        add(by_team[team], rec, cost); add(by_model[model], rec, cost)
        user_days[user].add(day); user_team[user] = team; team_users[team].add(user)
        heatmap[rec["w"]][rec["h"]] += 1
        for tool in tools:
            tool_calls[tool] += 1
            tool_users[tool].add(user)
        previous = last_seen.get(user)
        if previous is None or epoch - previous > SESSION_GAP:
            sessions[user] += 1
        last_seen[user] = epoch

    flat = [rec for _, _, rec, _, _ in records]
    days = sorted(by_day)
    daily = [dict(date=d, **by_day[d]) for d in days]
    cache_by_day = [{"date": d, "read": by_day[d]["cache_read_tokens"],
                     "write": by_day[d]["cache_write_tokens"],
                     "fresh": by_day[d]["input_tokens"]} for d in days]

    total_cost = sum(a["cost"] for a in by_day.values())
    users_out = sorted(
        [dict(user=u, team=user_team[u], sessions=sessions[u], active_days=len(user_days[u]),
              cost_per_active_day=by_user[u]["cost"] / len(user_days[u]), **by_user[u])
         for u in by_user], key=lambda x: -x["cost"])
    teams_out = sorted(
        [dict(team=t, users=len(team_users[t]),
              cost_share=(by_team[t]["cost"] / total_cost) if total_cost else 0.0, **by_team[t])
         for t in by_team], key=lambda x: -x["cost"])
    models_out = sorted([dict(model=m, **by_model[m]) for m in by_model], key=lambda x: -x["cost"])
    tools_out = [{"tool": t, "calls": c, "users": len(tool_users[t])}
                 for t, c in tool_calls.most_common()]

    fresh_in = sum(a["input_tokens"] for a in by_day.values())
    cached = sum(a["cache_read_tokens"] + a["cache_write_tokens"] for a in by_day.values())
    input_side = fresh_in + cached

    return {
        "meta": {"generated_at": generated_at.isoformat(), "account_id": account_id,
                 "bucket": BUCKET, "timezone": str(TZ),
                 "session_gap_minutes": SESSION_GAP // 60},
        "excluded": {"raw_invocations": raw_count, "kept_invocations": len(flat),
                     "no_team_tag": untagged, "credential_pings": pings, "errored": errored},
        "kpis": {"total_cost": total_cost, "total_calls": len(flat),
                 "active_users": len(by_user), "active_teams": len(by_team),
                 "total_sessions": sum(sessions.values()),
                 "avg_cost_per_user": (total_cost / len(by_user)) if by_user else 0.0,
                 "cache_share_of_tokens": (cached / input_side) if input_side else 0.0,
                 "pilot_start": days[0] if days else None,
                 "pilot_end": days[-1] if days else None},
        "records": flat, "daily": daily, "users": users_out, "teams": teams_out,
        "models": models_out, "tools": tools_out, "cache_by_day": cache_by_day,
        "cost_anatomy": {"input": sum(a["input_cost"] for a in by_day.values()),
                         "output": sum(a["output_cost"] for a in by_day.values()),
                         "cache_read": sum(a["cache_read_cost"] for a in by_day.values()),
                         "cache_write": sum(a["cache_write_cost"] for a in by_day.values())},
        "heatmap": heatmap,
    }


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="dashboard.html")
    ap.add_argument("-t", "--template", default="dashboard_template.html")
    ap.add_argument("--data-out", help="also write the raw dashboard payload as JSON")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    template = Path(args.template)
    if not template.is_absolute():
        template = here / template

    cfg = Config(max_pool_connections=WORKERS + 16,
                 retries={"max_attempts": 10, "mode": "adaptive"})
    session = boto3.session.Session()
    account_id = session.client("sts", config=cfg).get_caller_identity()["Account"]

    print("Reading IAM user Team tags...", file=sys.stderr)
    teams = load_teams(session.client("iam", config=cfg))
    print(f"  {len(teams)} tagged users across "
          f"{len(set(teams.values()))} teams", file=sys.stderr)

    s3 = session.client("s3", config=cfg)
    print(f"Listing s3://{BUCKET}/{LOG_PREFIX} ...", file=sys.stderr)
    keys = log_keys(s3)
    print(f"  {len(keys)} log files", file=sys.stderr)

    invocations = fetch_invocations(session, cfg, keys)
    print(f"  {len(invocations)} invocations", file=sys.stderr)

    generated_at = dt.datetime.now(dt.timezone.utc)
    data = build_data(invocations, teams, account_id, generated_at)

    payload = json.dumps(data, separators=(",", ":"))
    out = Path(args.out)
    if not out.is_absolute():
        out = here / out
    out.write_text(template.read_text().replace("__DATA__", payload))

    if args.data_out:
        Path(args.data_out).write_text(payload)

    k = data["kpis"]
    print(f"\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    print(f"  {k['pilot_start']} -> {k['pilot_end']}  |  ${k['total_cost']:,.2f}  |  "
          f"{k['total_calls']:,} calls  |  {k['active_users']} users  |  "
          f"{k['active_teams']} teams  |  {k['total_sessions']} sessions", file=sys.stderr)


if __name__ == "__main__":
    main()
