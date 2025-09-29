# Repository Guidelines

## Project Structure & Module Organization
Entry scripts sit in the repo root: `connect.py` onboards Gmail, `github_connect.py` links GitHub, `google_docs_connect.py` handles Google Docs, and `agent.py` replays stored identities. Keep helpers near these scripts until we extract a `src/` package. If you introduce modules, follow `src/<feature>.py` and place integration tests in `tests/`. Environment defaults live in `.env.example`, but each run resolves settings from your local `.env` via `python-dotenv`.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: set up a local virtualenv.
- `pip install composio openai python-dotenv`: install dependencies; refresh `requirements.txt` when the list changes.
- `python connect.py`: run Gmail OAuth, store a UUID, and send the canned test mail (uses the full Gmail toolkit).
- `python github_connect.py [uuid]`: run or reuse GitHub OAuth; positional/flagged UUIDs reuse the same identity and expose every GitHub tool.
- `python google_docs_connect.py [uuid]`: authorize Google Docs or reuse a stored UUID; the whole Docs toolkit becomes available automatically.
- `python agent.py <uuid> ["prompt"]`: reuse a UUID and invoke all connected toolkits (Gmail, GitHub, Google Docs, etc.); supply an optional quoted prompt or omit it to send the default email.

## Coding Style & Naming Conventions
Work in Python 3.11+, follow PEP 8, and favor explicit imports (`from composio import Composio`). Constants stay upper snake case; environment keys remain uppercase. Run `ruff check .` before submitting; if formatting drifts, apply `ruff format` or `black` to the touched files only.

## Testing Guidelines
We do not yet ship automated tests. When altering flows, add `pytest` suites under `tests/` using `test_<module>.py`. Mock external APIs (Composio, OpenAI) with `pytest-mock` to avoid real calls. Each feature should cover a happy path and a notable failure. Target 80% statement coverage with `pytest --cov=connect`, and document manual validation steps in PRs until coverage improves.

## Commit & Pull Request Guidelines
History favors concise imperative subjects (`Add Google Docs connector`). Keep the first line under 60 characters and add detail in the body when useful. PRs should link issues, describe the behavior change, list the commands you ran (OAuth flows, agents), and attach screenshots or logs for user-facing effects. Request review and wait for CI once available.

## Environment & Secrets
Copy `.env.example` to `.env` and fill `COMPOSIO_API_KEY`, `OPENAI_API_KEY`, `COMPOSIO_AUTH_CONFIG_ID`, `GITHUB_AUTH_CONFIG_ID`, `GOOGLE_DOCS_AUTH_CONFIG_ID`, and `EXTERNAL_USER_ID`. Optional overrides like `OPENAI_MODEL`, `EMAIL_SUBJECT`, `EMAIL_BODY`, `GITHUB_TASK`, and `GOOGLE_DOCS_TASK` fine-tune LLM behaviour. Never commit populated `.env` files. When demoing, scope each connected account to the minimum permissions required and rotate shared credentials often.
# Agents-using-Composio-MCP-tools
