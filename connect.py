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

# Prepare Composio runtime: cache dir + suppress telemetry thread tracebacks
_cache_dir = os.getenv("COMPOSIO_CACHE_DIR")
if not _cache_dir:
    from pathlib import Path as _P
    _local = _P(".composio_cache").resolve()
    _local.mkdir(parents=True, exist_ok=True)
    os.environ["COMPOSIO_CACHE_DIR"] = str(_local)

def _thread_hook(args: threading.ExceptHookArgs):
    try:
        tb = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    except Exception:
        tb = ""
    name = getattr(args.thread, "name", "")
    if "_thread_loop" in name and "composio/core/models/_telemetry" in tb:
        return
    sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)

threading.excepthook = _thread_hook

from composio import Composio
from openai import OpenAI
load_dotenv()

DEFAULT_OUTPUT_PATH = Path(
    os.getenv("ACCOUNT_OUTPUT_PATH", "outbox/connected_accounts.jsonl")
)
GMAIL_TOOLKITS = ["GMAIL"]
DEFAULT_SUBJECT = "Hello from composio 👋🏻"
DEFAULT_BODY = (
    "Congratulations on sending your first email using AI Agents and Composio!"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to Composio and send a test Gmail message."
    )
    parser.add_argument(
        "--record-id",
        help=(
            "Reuse an existing UUID from outbox/connected_accounts.jsonl to send an email "
            "without rerunning OAuth."
        ),
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} in environment")
    return value


def load_record(record_id: str, output_path: Path) -> dict[str, Any]:
    if not output_path.exists():
        raise RuntimeError(f"No stored connections found at {output_path}")

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record_id") == record_id:
                return record

    raise RuntimeError(f"Record {record_id} not found in {output_path}")


def clean_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [value for value in raw if isinstance(value, str) and value]
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return []


def main() -> None:
    args = parse_args()

    composio_api_key = require_env("COMPOSIO_API_KEY")
    openai_api_key = require_env("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    email_subject = os.getenv("EMAIL_SUBJECT", DEFAULT_SUBJECT)
    email_body = os.getenv("EMAIL_BODY", DEFAULT_BODY)
    email_recipient_env = os.getenv("TEST_EMAIL_RECIPIENT")

    composio = Composio(api_key=composio_api_key)
    openai_client = OpenAI(api_key=openai_api_key)

    if args.record_id:
        record = load_record(args.record_id, DEFAULT_OUTPUT_PATH)
        record_id = record["record_id"]
        external_user_id = record["user_id"]
        selected_toolkits = clean_list(record.get("toolkits")) or GMAIL_TOOLKITS
        account_id = record.get("connected_account_id", "<unknown>")
        print(
            "Reusing stored connection {account} for user {user}".format(
                account=account_id,
                user=external_user_id,
            )
        )
    else:
        auth_config_id = require_env("COMPOSIO_AUTH_CONFIG_ID")
        external_user_id = require_env("EXTERNAL_USER_ID")
        selected_toolkits = GMAIL_TOOLKITS

        connection_request = composio.connected_accounts.initiate(
            user_id=external_user_id,
            auth_config_id=auth_config_id,
        )

        redirect_url = connection_request.redirect_url
        print(f"Please authorize the app by visiting this URL: {redirect_url}")

        connected_account = connection_request.wait_for_connection()
        print(
            "Connection established successfully! Connected account id: "
            f"{connected_account.id}"
        )

        record_id = str(uuid.uuid4())
        record = {
            "record_id": record_id,
            "service": "gmail",
            "connected_account_id": connected_account.id,
            "user_id": external_user_id,
            "auth_config_id": auth_config_id,
            "toolkits": selected_toolkits,
            "stored_at": datetime.utcnow().isoformat() + "Z",
        }
        DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEFAULT_OUTPUT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        print(
            "Stored connection reference {record} for account {account} at {path}".format(
                record=record_id,
                account=connected_account.id,
                path=DEFAULT_OUTPUT_PATH,
            )
        )

    email_recipient = email_recipient_env or external_user_id
    tools = composio.tools.get(user_id=external_user_id, toolkits=selected_toolkits)

    user_prompt = (
        f"Send an email to {email_recipient} with the subject '{email_subject}' and the body "
        f"'{email_body} Tracking ID: {record_id}'."
    )

    response = openai_client.chat.completions.create(
        model=model_name,
        tools=tools,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
    )

    result = composio.provider.handle_tool_calls(
        response=response, user_id=external_user_id
    )
    print(result)
    print("Email sent successfully!")


if __name__ == "__main__":
    main()
