import argparse
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from composio import Composio
from dotenv import load_dotenv


load_dotenv()

DEFAULT_OUTPUT_PATH = Path(os.getenv("ACCOUNT_OUTPUT_PATH", "outbox/connected_accounts.jsonl"))

# Try common toolkit slugs for Google Calendar
GOOGLE_CALENDAR_TOOLKIT_CANDIDATES: list[str] = [
    "GOOGLECALENDAR",
    "GOOGLE_CALENDAR",
    "CALENDAR",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authorize Google Calendar via Composio and store the connection reference."
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


def _model_dump(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            pass
    return {"value": str(obj)}


def _extract_tool_name(tool: Any) -> Optional[str]:
    payload = _model_dump(tool)
    fn = payload.get("function") if isinstance(payload, dict) else None
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn.get("name")
    if isinstance(payload.get("name"), str):
        return payload.get("name")
    return None


def _unwrap_payload(response: Any) -> Dict[str, Any]:
    payload = _model_dump(response)
    if isinstance(payload, dict):
        for key in ("result", "data", "output"):
            if isinstance(payload.get(key), (dict, list)):
                return payload.get(key)
    return payload


def _find_calendar_record(records: List[Dict[str, Any]], record_id: str) -> Optional[Dict[str, Any]]:
    for rec in records:
        if rec.get("record_id") == record_id and rec.get("service") == "google_calendar":
            return rec
    return None


def update_record_timezone(path: Path, record_id: str, tz: str) -> None:
    # Load all, update in-memory, write back atomically
    all_recs: List[Dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    updated = False
    for rec in all_recs:
        if rec.get("record_id") == record_id and rec.get("service") == "google_calendar":
            rec["timezone"] = tz
            updated = True
    if not updated:
        # Append a minimal supplement record
        all_recs.append({
            "record_id": record_id,
            "service": "google_calendar",
            "timezone": tz,
        })
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for rec in all_recs:
            handle.write(json.dumps(rec) + "\n")
    tmp.replace(path)


def fetch_timezone_via_tools(composio: Composio, user_id: str, connected_account_id: Optional[str]) -> Optional[str]:
    # Collect tools from candidate slugs
    tools: List[Any] = []
    for slug in GOOGLE_CALENDAR_TOOLKIT_CANDIDATES:
        try:
            got = composio.tools.get(user_id=user_id, toolkits=[slug])
        except Exception:
            got = []
        if got:
            tools.extend(got)
    if not tools:
        return None

    tool_names = [(t, (_extract_tool_name(t) or "").upper()) for t in tools]

    def try_execute(t: Any, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        name = _extract_tool_name(t)
        if not name:
            return None
        try:
            resp = composio.tools.execute(
                slug=name,
                arguments=arguments,
                user_id=user_id,
                connected_account_id=connected_account_id,
            )
        except Exception:
            return None
        return _unwrap_payload(resp)

    # 1) Settings API: users/me/settings/timezone
    for t, up in tool_names:
        if "SETTING" in up and "GET" in up and ("TIMEZONE" in up or True):
            for key in ("setting", "setting_id", "settingId", "name"):
                args = {key: "timezone"}
                data = try_execute(t, args)
                if isinstance(data, dict):
                    # Common shapes: {value: "Asia/Kolkata"} or {timeZone: "..."} or {timezone: "..."}
                    for tz_key in ("value", "timeZone", "timezone", "tz"):
                        tz_val = data.get(tz_key)
                        if isinstance(tz_val, str) and tz_val:
                            return tz_val

    # 2) calendars.get('primary') or calendarList.get('primary')
    for t, up in tool_names:
        if "GET" in up and "EVENT" not in up and ("CALENDAR" in up or "CALENDARLIST" in up):
            for key in ("calendar_id", "calendarId", "id"):
                data = try_execute(t, {key: "primary"})
                if isinstance(data, dict):
                    tz_val = data.get("timeZone") or data.get("timezone")
                    if isinstance(tz_val, str) and tz_val:
                        return tz_val

    # 3) calendarList.list: find primary
    for t, up in tool_names:
        if "LIST" in up and "CALENDAR" in up:
            data = try_execute(t, {})
            if isinstance(data, dict):
                items = data.get("items") or data.get("calendars") or data.get("results")
                if isinstance(items, list):
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        if it.get("primary") is True or it.get("id") == "primary":
                            tz_val = it.get("timeZone") or it.get("timezone")
                            if isinstance(tz_val, str) and tz_val:
                                return tz_val
    return None


def main() -> None:
    args = parse_args()

    composio_api_key = require_env("COMPOSIO_API_KEY")
    auth_config_id = os.getenv("GOOGLE_CALENDAR_AUTH_CONFIG_ID")

    record_id_positional = args.record_id
    record_id_flag = getattr(args, "record_id_flag", None)
    if record_id_positional and record_id_flag and record_id_positional != record_id_flag:
        raise RuntimeError("Conflicting record IDs provided; supply only one value.")
    provided_record_id = record_id_positional or record_id_flag

    if provided_record_id:
        existing = load_records(provided_record_id, DEFAULT_OUTPUT_PATH)
        record_id = provided_record_id
        external_user_id = select_user_id(existing) or require_env("EXTERNAL_USER_ID")
        # Check if a Google Calendar connection already exists for this record
        reuse = any(rec.get("service") == "google_calendar" for rec in existing)
        if not reuse and not auth_config_id:
            auth_config_id = require_env("GOOGLE_CALENDAR_AUTH_CONFIG_ID")
    else:
        record_id = str(uuid.uuid4())
        external_user_id = require_env("EXTERNAL_USER_ID")
        if not auth_config_id:
            auth_config_id = require_env("GOOGLE_CALENDAR_AUTH_CONFIG_ID")
        existing = []
        reuse = False

    composio = Composio(api_key=composio_api_key)

    connected_account_id: Optional[str] = None
    if not reuse:
        # Start OAuth for Calendar
        connection_request = composio.connected_accounts.initiate(
            user_id=external_user_id,
            auth_config_id=auth_config_id,
        )
        print(
            "Please authorize Google Calendar access by visiting this URL: "
            f"{connection_request.redirect_url}"
        )
        connected_account = connection_request.wait_for_connection()
        print(
            "Google Calendar connection established successfully! Connected account id: "
            f"{connected_account.id}"
        )

        connected_account_id = connected_account.id
        record = {
            "record_id": record_id,
            "service": "google_calendar",
            "connected_account_id": connected_account_id,
            "user_id": external_user_id,
            "auth_config_id": auth_config_id,
            "toolkits": GOOGLE_CALENDAR_TOOLKIT_CANDIDATES,
            "stored_at": datetime.utcnow().isoformat() + "Z",
        }
        write_record(record, DEFAULT_OUTPUT_PATH)
        print(
            "Stored Google Calendar connection {record} for user {user} at {path}".format(
                record=record_id,
                user=external_user_id,
                path=DEFAULT_OUTPUT_PATH,
            )
        )
    else:
        print(
            "Reusing existing Google Calendar connection for record {record}".format(
                record=record_id,
            )
        )
        cal_rec = _find_calendar_record(existing, record_id)
        if isinstance(cal_rec, dict):
            connected_account_id = cal_rec.get("connected_account_id")

    # Try to fetch tools to verify the connection and pick a working toolkit slug
    tools = []
    used_slug = None
    for slug in GOOGLE_CALENDAR_TOOLKIT_CANDIDATES:
        try:
            tools = composio.tools.get(user_id=external_user_id, toolkits=[slug])
        except Exception:
            tools = []
        if tools:
            used_slug = slug
            break

    if used_slug:
        print(f"[i] Calendar tools resolved via '{used_slug}': {len(tools)} tools available")
    else:
        print("[warn] No Google Calendar tools returned; verify toolkit slug and permissions in Composio.")

    # Try to discover timezone via OAuth and store it on the record
    tz_val = fetch_timezone_via_tools(composio, external_user_id, connected_account_id)
    if tz_val:
        update_record_timezone(DEFAULT_OUTPUT_PATH, record_id, tz_val)
        print(f"[i] Discovered calendar timezone: {tz_val} (saved to record)")
    else:
        print("[warn] Could not determine calendar timezone from API; set LOCAL_TZ env as a fallback.")

    print("Google Calendar setup completed.")


if __name__ == "__main__":
    main()
