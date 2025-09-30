import json
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

from dotenv import load_dotenv

WORKDIR = Path(__file__).resolve().parent
DOTENV_PATH = WORKDIR / ".env"
if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH, override=False)
else:
    load_dotenv(override=False)

CACHE_DIR = Path(os.getenv("COMPOSIO_CACHE_DIR", WORKDIR / ".composio_cache")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("COMPOSIO_CACHE_DIR", str(CACHE_DIR))

def _prep_composio_env():
    cache_dir = os.getenv("COMPOSIO_CACHE_DIR")
    if not cache_dir:
        local = Path(".composio_cache").resolve()
        local.mkdir(parents=True, exist_ok=True)
        os.environ["COMPOSIO_CACHE_DIR"] = str(local)

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

_prep_composio_env()

from composio import Composio  # noqa: E402

DEFAULT_OUTPUT_PATH = Path(os.getenv("ACCOUNT_OUTPUT_PATH", WORKDIR / "outbox/connected_accounts.jsonl"))
DEFAULT_USER_ID = os.getenv("EXTERNAL_USER_ID")
SERVICE_TOOLKITS = {
    "gmail": ["GMAIL"],
    "github": ["GITHUB"],
    "google_docs": ["GOOGLEDOCS"],
    "notion": ["NOTION"],
}

# Some environments expose slightly different toolkit slugs; try fallbacks.
TOOLKIT_SYNONYMS: dict[str, list[str]] = {
    "GMAIL": ["GMAIL", "GOOGLE_GMAIL"],
    "GITHUB": ["GITHUB"],
    # Correct slug is GOOGLEDOCS; include older variants as fallbacks
    "GOOGLEDOCS": [
        "GOOGLEDOCS",
        "GOOGLE_DOCS",
        "GOOGLE_DOCUMENTS",
        "GOOGLE_DOC",
        "DOCS",
        "GOOGLE_DRIVE",  # Some Docs actions may be surfaced under Drive
        "GOOGLE",
    ],
    # Backward-compatibility: if older records still use GOOGLE_DOCS, try the same aliases
    "GOOGLE_DOCS": [
        "GOOGLEDOCS",
        "GOOGLE_DOCS",
        "GOOGLE_DOCUMENTS",
        "GOOGLE_DOC",
        "DOCS",
        "GOOGLE_DRIVE",
        "GOOGLE",
    ],
    # Google Calendar synonyms
    "GOOGLECALENDAR": [
        "GOOGLECALENDAR",
        "GOOGLE_CALENDAR",
        "CALENDAR",
    ],
    # Notion synonyms across environments
    "NOTION": [
        "NOTION",
        "NOTIONHQ",
        "NOTION_DB",
        "NOTIONDATABASES",
        "NOTION_DATABASE",
        "NOTIONDATABASE",
        "NOTION_PAGES",
        "NOTIONPAGES",
    ],
}


def canonicalize_toolkit_slug(slug: str) -> str:
    up = slug.upper()
    for canonical, aliases in TOOLKIT_SYNONYMS.items():
        if up == canonical or up in {a.upper() for a in aliases}:
            return canonical
    return up


def canonicalize_toolkits(slugs: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for s in slugs:
        can = canonicalize_toolkit_slug(s)
        if can not in seen:
            seen.add(can)
            result.append(can)
    return result


def debug_env(name: str) -> None:
    value = os.getenv(name)
    if value:
        print(f"[env] {name} is set")
    else:
        print(f"[env] {name} is MISSING")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        debug_env(name)
        raise RuntimeError(f"Missing {name} in environment")
    return value


def clean_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [value for value in raw if isinstance(value, str) and value]
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return []


def load_records(path: Path, record_id: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record_id and record.get("record_id") != record_id:
                continue
            records.append(record)
    return records


def collect_toolkits(records: list[dict[str, Any]]) -> list[str]:
    toolkits: set[str] = set()
    for record in records:
        service = record.get("service")
        cleaned = clean_list(record.get("toolkits"))
        if cleaned:
            toolkits.update(cleaned)
        elif service in SERVICE_TOOLKITS:
            toolkits.update(SERVICE_TOOLKITS[service])
    if not toolkits:
        for service_toolkits in SERVICE_TOOLKITS.values():
            toolkits.update(service_toolkits)
    return sorted(toolkits)


def summarize_records(records: list[dict[str, Any]]) -> str:
    if not records:
        return "(no stored connections)"
    summary_parts: list[str] = []
    for record in records:
        rid = record.get("record_id", "<unknown>")
        service = record.get("service", "unspecified")
        toolkits = ",".join(clean_list(record.get("toolkits"))) or "<none>"
        summary_parts.append(f"{rid}:{service}[{toolkits}]")
    return "; ".join(summary_parts)


def _to_payload(tool: Any) -> Dict[str, Any]:
    """Convert various tool objects into a plain dict."""
    if isinstance(tool, dict):
        return tool
    if hasattr(tool, "model_dump"):
        try:
            return tool.model_dump()
        except Exception:
            pass
    if hasattr(tool, "dict"):
        try:
            return tool.dict()
        except Exception:
            pass
    if hasattr(tool, "__dict__"):
        try:
            return dict(tool.__dict__)
        except Exception:
            pass
    return {}


def _extract_name_toolkit_description(payload: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract (name, toolkit, description) including OpenAI function spec.

    - Prefer `payload["function"]["name"|"description"]` when present.
    - Fallback to common top-level fields.
    - Infer toolkit from name prefix like GMAIL_SEND_MESSAGE if missing.
    """
    function = payload.get("function") if isinstance(payload, dict) else None
    name = None
    description = None
    if isinstance(function, dict):
        name = function.get("name") or None
        description = function.get("description") or None

    name = (
        name
        or payload.get("name")
        or payload.get("tool_id")
        or payload.get("slug")
        or payload.get("display_name")
        or payload.get("id")
        or payload.get("identifier")
    )

    raw_toolkit = (
        payload.get("toolkit")
        or payload.get("toolkit_id")
        or payload.get("toolkit_slug")
        or (function.get("toolkit") if isinstance(function, dict) else None)
    )
    toolkit = None
    if isinstance(raw_toolkit, dict):
        toolkit = raw_toolkit.get("slug") or raw_toolkit.get("name")
    elif isinstance(raw_toolkit, str):
        toolkit = raw_toolkit

    # Infer toolkit from tool name tokens
    # Examples:
    # - GOOGLEDOCS_CREATE_DOC    -> GOOGLEDOCS
    # - GOOGLE_DOCS_CREATE_DOC   -> GOOGLEDOCS (normalized)
    # - GMAIL_SEND_MESSAGE       -> GMAIL
    if not toolkit and isinstance(name, str):
        tokens = name.split("_")
        if tokens:
            first = tokens[0].upper()
            second = tokens[1].upper() if len(tokens) > 1 else None
            if first == "GOOGLEDOCS":
                toolkit = "GOOGLEDOCS"
            elif first == "GOOGLE" and second == "DOCS":
                toolkit = "GOOGLEDOCS"
            else:
                toolkit = first

    if not isinstance(name, str) or not name.strip():
        name = "<unnamed>"
    if not isinstance(toolkit, str) or not toolkit.strip():
        toolkit = "<unknown toolkit>"
    if not isinstance(description, str):
        description = payload.get("description") or payload.get("summary") or ""

    return name, toolkit, description


def describe_tool(tool: Any) -> tuple[str, str, str]:
    # Quick path if attributes are directly available
    attr_name = getattr(tool, "name", None)
    attr_toolkit = getattr(tool, "toolkit", None)
    attr_description = getattr(tool, "description", None)
    if isinstance(attr_name, str) and isinstance(attr_toolkit, str):
        return attr_name, attr_toolkit, attr_description or ""

    payload = _to_payload(tool)
    return _extract_name_toolkit_description(payload)


def main() -> None:
    user_id = DEFAULT_USER_ID
    if not user_id:
        user_id = input("Enter the user UUID to inspect tools for: ").strip()
    if not user_id:
        raise RuntimeError("User ID is required to list tools.")

    specific_record = os.getenv("TOOLS_RECORD_ID") or None
    records = load_records(DEFAULT_OUTPUT_PATH, record_id=specific_record)
    toolkits = collect_toolkits(records)

    debug_env("COMPOSIO_API_KEY")
    composio_api_key = require_env("COMPOSIO_API_KEY")
    composio = Composio(api_key=composio_api_key)

    print(f"Fetching tools for user {user_id}...")
    if specific_record:
        print(f"Using record filter: {specific_record}")
    toolkits = canonicalize_toolkits(toolkits)
    print(f"Derived toolkits: {', '.join(toolkits)}")
    if records:
        print(f"Known connections: {summarize_records(records)}\n")
    else:
        print("No stored connections found locally; using default toolkit list.\n")

    # Fetch per-toolkit (with synonym fallbacks) to avoid API quirks
    tools_by_toolkit: dict[str, list[Any]] = {}
    for tk in toolkits:
        slugs_to_try = TOOLKIT_SYNONYMS.get(tk.upper(), [tk])
        tk_tools: list[Any] = []
        used_slug = None
        for slug in slugs_to_try:
            try:
                tk_tools = composio.tools.get(user_id=user_id, toolkits=[slug])
            except Exception as exc:
                raise RuntimeError(f"Failed to fetch tools for {slug}: {exc}") from exc
            if tk_tools:
                used_slug = slug
                break
        if not tk_tools:
            print(f"[warn] No tools returned for {tk} (tried: {', '.join(slugs_to_try)})")
            continue
        if used_slug and used_slug != tk:
            print(f"[info] Loaded {len(tk_tools)} tools for {tk} via alias '{used_slug}'")
        tools_by_toolkit[tk] = tk_tools

    total = sum(len(v) for v in tools_by_toolkit.values())
    if total == 0:
        print("No tools returned for this user.")
        return

    print("Tools available:")
    for tk in toolkits:
        group = tools_by_toolkit.get(tk, [])
        if not group:
            continue
        print(f"\n[{tk}] ({len(group)} tools)")
        for tool in group:
            name, toolkit, description = describe_tool(tool)
            print(f"- {name} (toolkit: {toolkit})")
            if description:
                print(f"    {description}")


if __name__ == "__main__":
    main()
