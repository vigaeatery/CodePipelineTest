# Claude Code Pilot Register

A single-file HTML dashboard covering the City of Boston Claude Code pilot,
rebuilt from live AWS data by `generate_dashboard.py`.

## What it does

1. **Reads Bedrock model-invocation logs** from `s3://boston-claude-pilot-logs`
   (`AWSLogs/<account>/BedrockModelInvocationLogs/...`). Only the hourly log
   files are read; the `data/` prefix holding offloaded request bodies (~4 GB)
   is skipped, because every field the dashboard needs — timestamps, token
   counts, model IDs, caller identity and `tool_use` block names — is inline in
   the log line itself.
2. **Attributes each call to a team** by taking the IAM user from the
   invocation's `identity.arn` and reading that user's `Team` tag via
   `iam:ListUsers` / `iam:ListUserTags`.
3. **Aggregates and renders** the result into `dashboard_template.html`, writing
   `dashboard.html`.

No prompt or response text is read or stored — only token counts, timestamps,
model IDs and tool names.

## Usage

```bash
pip install boto3
python3 generate_dashboard.py            # -> dashboard.html
python3 generate_dashboard.py -o out.html --data-out payload.json
```

Requires credentials with `s3:GetObject`/`s3:ListBucket` on the log bucket and
`iam:ListUsers` / `iam:ListUserTags`. A full run reads ~10,700 log files in
about a minute.

## What gets excluded

Invocations are dropped in this order, and the counts are reported in the
dashboard footer:

| Exclusion | Why |
| --- | --- |
| `errorCode` present | Bedrock returned an error; no usage was billed |
| One-token `"."` prompt | Claude Code's credential-validation ping |
| IAM user has no `Team` tag | Service accounts, shared and test logins |

## Cost model

Spend is *estimated* from token counts at Bedrock on-demand list prices, in USD
per million tokens (see `rates()`):

| Family | Input | Output | Cache read | Cache write |
| --- | --- | --- | --- | --- |
| Opus (4.6 – 5) | 5.00 | 25.00 | 0.50 | 6.25 |
| Sonnet 5 | 2.00 | 10.00 | 0.20 | 2.50 |
| Sonnet (4.5, 4.6) | 3.00 | 15.00 | 0.30 | 3.75 |
| Haiku 4.5 | 1.00 | 5.00 | 0.10 | 1.25 |

Models outside this table (non-Anthropic models on the same account) are billed
at zero. Confirm against the AWS bill for exact figures.

## Other conventions

- **Sessions** are runs of one user's invocations with less than 30 minutes
  between consecutive calls.
- **Dates, weekdays and hours** are local `America/New_York`.
- **Model names** are normalized from the inference-profile ARN, e.g.
  `arn:...:inference-profile/us.anthropic.claude-opus-5:0` → `us.anthropic.claude-opus-5`.

## Files

| File | |
| --- | --- |
| `generate_dashboard.py` | The pipeline: S3 scan, IAM tag lookup, aggregation, render |
| `dashboard_template.html` | Presentation layer; `__DATA__` is the payload placeholder |
| `dashboard.html` | Generated output — open directly in a browser |
