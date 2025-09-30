import argparse
import json
import os
import sys
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure a writable Composio cache directory and suppress telemetry thread tracebacks
_cache_dir = os.getenv("COMPOSIO_CACHE_DIR")
if not _cache_dir:
    _local = Path(".composio_cache").resolve()
    _local.mkdir(parents=True, exist_ok=True)
    os.environ["COMPOSIO_CACHE_DIR"] = str(_local)

def _composio_thread_excepthook(args: threading.ExceptHookArgs):
    try:
        tb = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    except Exception:
        tb = ""
    name = getattr(args.thread, "name", "")
    if "_thread_loop" in name and "composio/core/models/_telemetry" in tb:
        return
    sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)

threading.excepthook = _composio_thread_excepthook

from composio import Composio
from openai import OpenAI


load_dotenv()

DEFAULT_OUTPUT_PATH = Path(
    os.getenv("ACCOUNT_OUTPUT_PATH", "outbox/connected_accounts.jsonl")
)

# Try common toolkit slugs for Notion
NOTION_TOOLKIT_CANDIDATES: list[str] = [
    "NOTION",
    "NOTIONHQ",
    "NOTION_DB",
    "NOTIONDATABASES",
]

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TASK = os.getenv(
    "NOTION_TASK",
    "What Notion actions can you perform for this user?",
)
PROMPT_TEMPLATE = (
    "{task} Capture the tracking ID {tracking_id} in any updates you write."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authorize Notion via Composio and run a simple tool-enabled task."
    )
    parser.add_argument(
        "record_id",
        nargs="?",
        help="UUID in outbox/connected_accounts.jsonl to reuse an existing agent identity.",
    )
    parser.add_argument(
        "--record-id",
        dest="record_id_flag",
        help="Optional flag alias for providing the stored UUID explicitly.",
    )
    parser.add_argument(
        "--task",
        help="Override the default Notion agent task.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to invoke (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=int(os.getenv("OAUTH_WAIT_SECONDS", "240")),
        help="How long to wait for OAuth to complete before timing out (default: 240).",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} in environment")
    return value


def clean_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [value for value in raw if isinstance(value, str) and value]
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return []


def write_record(record: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_records(record_id: str, path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    matches: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record_id") == record_id:
                matches.append(record)
    return matches


def select_user_id(records: list[dict[str, Any]]) -> str | None:
    for record in records:
        user_id = record.get("user_id")
        if isinstance(user_id, str) and user_id:
            return user_id
    return None


def select_toolkits(
    records: list[dict[str, Any]],
    *,
    service: str,
    fallback: list[str],
) -> list[str]:
    for record in records:
        if record.get("service") == service:
            cleaned = clean_list(record.get("toolkits"))
            if cleaned:
                return cleaned
    return fallback


def main() -> None:
    args = parse_args()

    composio_api_key = require_env("COMPOSIO_API_KEY")
    openai_api_key = require_env("OPENAI_API_KEY")
    auth_config_id = os.getenv("NOTION_AUTH_CONFIG_ID")

    record_id_positional = args.record_id
    record_id_flag = getattr(args, "record_id_flag", None)
    if record_id_positional and record_id_flag and record_id_positional != record_id_flag:
        raise RuntimeError("Conflicting record IDs provided; supply only one value.")
    provided_record_id = record_id_positional or record_id_flag

    if provided_record_id:
        existing_records = load_records(provided_record_id, DEFAULT_OUTPUT_PATH)
        record_id = provided_record_id
        external_user_id = select_user_id(existing_records) or require_env("EXTERNAL_USER_ID")
        requested_toolkits = select_toolkits(
            existing_records,
            service="notion",
            fallback=NOTION_TOOLKIT_CANDIDATES,
        )
        notion_record = next(
            (rec for rec in existing_records if rec.get("service") == "notion"),
            None,
        )
        reuse_connection = notion_record is not None
        if not reuse_connection and not auth_config_id:
            auth_config_id = require_env("NOTION_AUTH_CONFIG_ID")
    else:
        record_id = str(uuid.uuid4())
        external_user_id = require_env("EXTERNAL_USER_ID")
        requested_toolkits = NOTION_TOOLKIT_CANDIDATES
        if not auth_config_id:
            auth_config_id = require_env("NOTION_AUTH_CONFIG_ID")
        existing_records = []
        reuse_connection = False

    composio = Composio(api_key=composio_api_key)

    if not reuse_connection:
        connection_request = composio.connected_accounts.initiate(
            user_id=external_user_id,
            auth_config_id=auth_config_id,
        )
        redirect_url = connection_request.redirect_url
        print(
            "Please authorize Notion access by visiting this URL: {url}\n"
            "[i] Connection session id: {sid}"
            .format(url=redirect_url, sid=getattr(connection_request, "id", "<unknown>"))
        )
        try:
            connected_account = connection_request.wait_for_connection(timeout=args.wait_seconds)
        except Exception as e:
            # Provide a clearer hint on common causes like timeouts
            msg = str(e)
            print(
                "[error] Waiting for Notion OAuth confirmation failed: {msg}\n"
                "- Ensure you completed the browser authorization flow.\n"
                "- If it timed out, rerun with a larger wait: --wait-seconds 600.\n"
                "- Also verify NOTION_AUTH_CONFIG_ID points to a valid Composio auth config (usually starts with 'ac_')."
                .format(msg=msg)
            )
            raise
        print(
            "Notion connection established successfully! Connected account id: "
            f"{connected_account.id}"
        )
        record = {
            "record_id": record_id,
            "service": "notion",
            "connected_account_id": connected_account.id,
            "user_id": external_user_id,
            "auth_config_id": auth_config_id,
            "toolkits": requested_toolkits,
            "stored_at": datetime.utcnow().isoformat() + "Z",
        }
        write_record(record, DEFAULT_OUTPUT_PATH)
        print(
            "Stored Notion connection {record} for user {user} at {path}".format(
                record=record_id,
                user=external_user_id,
                path=DEFAULT_OUTPUT_PATH,
            )
        )
    else:
        print(
            "Reusing existing Notion connection for record {record}".format(
                record=record_id,
            )
        )

    # Attempt to resolve available Notion tools using candidate slugs
    tools = []
    used_slug = None
    for slug in NOTION_TOOLKIT_CANDIDATES:
        try:
            tools = composio.tools.get(user_id=external_user_id, toolkits=[slug])
        except Exception:
            tools = []
        if tools:
            used_slug = slug
            break

    if used_slug:
        print(f"[i] Notion tools resolved via '{used_slug}': {len(tools)} tools available")
    else:
        print(
            "[warn] No Notion tools returned; verify toolkit slug and permissions in Composio."
        )

    # Optional demo: invoke model with tools enabled
    openai_client = OpenAI(api_key=openai_api_key)
    task = args.task or DEFAULT_TASK
    user_prompt = PROMPT_TEMPLATE.format(task=task, tracking_id=record_id)

    response = openai_client.chat.completions.create(
        model=args.model,
        tools=tools,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Notion integration agent. Choose actions responsibly and "
                    "include the provided tracking ID in any updates."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    result = composio.provider.handle_tool_calls(
        response=response,
        user_id=external_user_id,
    )
    print(json.dumps(result, indent=2))
    print("Notion agent task completed.")


if __name__ == "__main__":
    main()
