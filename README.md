# Agents using Composio MCP Tools

Automation scripts and agents that use Composio’s MCP tools with OpenAI to operate on Gmail, Google Docs/Drive, GitHub, and Google Calendar. OAuth connections are stored locally so you can reuse identities across runs.

## File Structure
```
.
├── agent.py                     # Google Docs orchestrator agent (LangChain + Composio)
├── calendar_agent.py            # Google Calendar agent (optimized, LangChain)
├── connect.py                   # Gmail: connect via OAuth and send a test email
├── github_connect.py            # GitHub: connect and run a simple task via tools
├── google_calendar_connect.py   # Google Calendar: connect and save timezone to record
├── google_docs_connect.py       # Google Docs: connect and run a simple task
├── mail_agent.py                # Gmail orchestrator agent (LangChain)
├── tools.py                     # Inspect available tools for a user
├── outbox/
│   └── connected_accounts.jsonl # Per‑line records of connected accounts and UUIDs
├── requirements.txt
├── .env                         # Local environment (ignored)
├── .gitignore
└── README.md
```

## Prerequisites
- Python 3.11+
- OpenAI API key and Composio API key

## Setup
- Create and activate a virtualenv:
  - `python3 -m venv .venv && source .venv/bin/activate`
- Install dependencies:
  - `pip install -r requirements.txt`
- Create `.env` in the repo root with at least:
  ```env
  # Required
  COMPOSIO_API_KEY=...
  OPENAI_API_KEY=...
  EXTERNAL_USER_ID=you@example.com            # or any stable user UUID

  # Per‑service Auth Config IDs (from Composio)
  COMPOSIO_AUTH_CONFIG_ID=...                 # Gmail (used by connect.py)
  GITHUB_AUTH_CONFIG_ID=...
  GOOGLE_DOCS_AUTH_CONFIG_ID=...
  GOOGLE_CALENDAR_AUTH_CONFIG_ID=...

  # Optional
  OPENAI_MODEL=gpt-4o-mini
  ACCOUNT_OUTPUT_PATH=outbox/connected_accounts.jsonl
  TEST_EMAIL_RECIPIENT=you@example.com        # default falls back to EXTERNAL_USER_ID
  EMAIL_SUBJECT=Hello from Composio
  EMAIL_BODY=Congrats on your first MCP email!
  EMAIL_SUMMARY_DAYS=7
  EMAIL_MAX_RESULTS=25
  CALENDAR_SUMMARY_DAYS=7
  CALENDAR_MAX_RESULTS=25
  CALENDAR_TOOLS_CACHE_TTL_SECONDS=86400
  ```

## First‑Time Connections (OAuth)
Records are appended to `outbox/connected_accounts.jsonl` so you can reuse them later with a `record_id`.

- Gmail: `python connect.py`
  - Prints an OAuth URL; approve access, then it sends a test email and writes a record with a new `record_id`.
  - Reuse later: `python connect.py --record-id <record_uuid>`

- GitHub: `python github_connect.py [--record-id <record_uuid>] [--task "..."]`
  - Example: `python github_connect.py --task "List my repos"`

- Google Docs: `python google_docs_connect.py [--record-id <record_uuid>] [--task "..."]`
  - Example: `python google_docs_connect.py --task "Create a doc named 'Demo'"`

- Google Calendar: `python google_calendar_connect.py [--record-id <record_uuid>]`
  - Discovers and stores your Calendar timezone when possible.

## Running Agents
- Gmail agent: `python mail_agent.py <user_or_record_uuid> "Summarize emails from last 7 days" [--model gpt-4o-mini]`
- Google Docs agent: `python agent.py <user_or_record_uuid> "Create a Google Doc titled 'Weekly Summary'" [--model gpt-4o-mini]`
- Calendar agent: `python calendar_agent.py <user_or_record_uuid> "List my meetings next week" [--fast] [--debug]`

Notes
- For all agents, you may pass either the Composio `user_id` (EXTERNAL_USER_ID) or a stored `record_id`. If a `record_id` is provided, the scripts look up the underlying `user_id` and any connected account.
- Tool availability comes from Composio; if zero tools are returned, verify the correct Auth Config ID and granted scopes in your Composio dashboard.

## Utilities
- List tools for a user: `python tools.py`
  - Honors `EXTERNAL_USER_ID` from `.env`. Set `TOOLS_RECORD_ID=<record_uuid>` to scope to a specific record.

## Troubleshooting
- Missing env var: the scripts fail fast with a message like “Missing COMPOSIO_API_KEY in environment”. Add it to `.env` and retry.
- No tools returned: ensure the relevant `*_AUTH_CONFIG_ID` is set and the OAuth flow completed successfully. Some environments expose different toolkit slugs; the scripts try common aliases.
- Calendar timezone not set: rerun `python google_calendar_connect.py` to attempt discovery, or set a timezone hint in your prompt when using the agent.
- Reusing identities: pass `--record-id <record_uuid>` to the connector scripts, or pass `<record_uuid>` to the agents.

## Notes
- Connection records are stored in `outbox/connected_accounts.jsonl`. One JSON object per line with fields like `record_id`, `service`, `connected_account_id`, and `user_id`.
- `.env` is ignored by git by default; keep secrets out of version control.
