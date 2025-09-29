import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from composio import Composio

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.prompts.string import jinja2_formatter
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts.string import jinja2_formatter


load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_RECORD_PATH = Path(os.getenv("ACCOUNT_OUTPUT_PATH", "outbox/connected_accounts.jsonl"))
CALENDAR_MAX_RESULTS = int(os.getenv("CALENDAR_MAX_RESULTS", "25"))
DEFAULT_WINDOW_DAYS = int(os.getenv("CALENDAR_SUMMARY_DAYS", "7"))

# Toolkit candidates to fetch Calendar tools regardless of slug variants
GOOGLE_CALENDAR_TOOLKIT_CANDIDATES: List[str] = [
    "GOOGLECALENDAR",
    "GOOGLE_CALENDAR",
    "CALENDAR",
]

# Cache for tool function specs (speeds up repeated runs)
TOOLS_CACHE_DIR = (Path.cwd() / ".composio_cache").resolve()
TOOLS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TOOLS_CACHE_TTL_SEC = int(os.getenv("CALENDAR_TOOLS_CACHE_TTL_SECONDS", "86400"))  # 24h


def _tools_cache_path(user_id: str) -> Path:
    safe = user_id.replace("/", "_").replace(":", "_")
    return TOOLS_CACHE_DIR / f"calendar_tools_{safe}.json"


def _load_cached_functions(user_id: str) -> Optional[List[Dict[str, Any]]]:
    path = _tools_cache_path(user_id)
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age > TOOLS_CACHE_TTL_SEC:
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else None
    except Exception:
        return None


def _save_cached_functions(user_id: str, functions: List[Dict[str, Any]]) -> None:
    try:
        with _tools_cache_path(user_id).open("w", encoding="utf-8") as fh:
            json.dump(functions, fh)
    except Exception:
        pass


SYSTEM_PROMPT_JINJA = """
You are the Google Calendar Agent. Operate precisely, safely, and concisely.

Context
- Current time: {{ current_time }}
- Timezone hint: {{ tz_hint or "unknown" }}
- Max results: {{ max_results }}
- User ID: {{ user_id }}
- Connected account: {{ connected_account_id or "unset" }}

Mission
- Convert the user’s natural language into a normalized JSON plan (NIS: Normalized Input Schema).
- Discover the user’s true calendar timezone from Calendar APIs with high confidence.
- Convert NIS into tool-specific payloads and execute via Composio Calendar tools.
- Format a clean, user-facing Markdown answer with emojis. Never leak internal JSON or tool traces.

Timezone Discovery Protocol (CRITICAL)
1) Try, in order (stop at first success):
   a. users.settings.get("timezone") or equivalent “settings/timezone” tool
   b. calendarList.get("primary") or calendars.get("primary")
   c. calendarList.list → find item with primary=true or id="primary"
2) If a timezone is discovered, use it and discard {{ tz_hint }} if conflicting.
3) If no timezone can be discovered, do NOT guess. Ask one concise question:
   “Which timezone should I use (e.g., Asia/Kolkata)?” and stop.
4) Confidence must be ≥ 0.95 before proceeding. Otherwise, ask the user.

Time Construction Rules
- Always include timezone in timestamps:
  A) ISO-8601 with explicit offset, e.g., 2025-09-27T12:00:00+05:30
  B) Nested shape: start: {dateTime: "...+05:30", timeZone: "Asia/Kolkata"}
- Resolve relative phrases (today/tomorrow/this week/next week) to local calendar-day boundaries
  using the discovered timezone (not system).
- Defaults if omitted:
  • create_event duration: 30 minutes
  • list_events window: [now, now + {{ default_window_days }} days]
- All-day events: use date (YYYY-MM-DD), not dateTime.
- Free/busy: send explicit time_min/time_max with timezone offset.

Normalized Input Schema (NIS)
// First produce this internal JSON (do not show to the user):
{
  "action": "<one of: list_events | free_busy | get_event | create_event | update_event | delete_event>",
  "params": {
    // Common
    "calendar_id": "primary" | "<id>",

    // list_events
    "time_min": "YYYY-MM-DDTHH:MM:SS+HH:MM",
    "time_max": "YYYY-MM-DDTHH:MM:SS+HH:MM",
    "query": "<text>",
    "max_results": {{ max_results }},

    // free_busy
    "time_min": "YYYY-MM-DDTHH:MM:SS+HH:MM",
    "time_max": "YYYY-MM-DDTHH:MM:SS+HH:MM",

    // get_event
    "event_id": "<id>",

    // create_event
    "summary": "<title>",
    "description": "<text>",
    "start": { "dateTime": "YYYY-MM-DDTHH:MM:SS+HH:MM", "timeZone": "<IANA>" } | { "date": "YYYY-MM-DD" },
    "end":   { "dateTime": "YYYY-MM-DDTHH:MM:SS+HH:MM", "timeZone": "<IANA>" } | { "date": "YYYY-MM-DD" },
    "location": "<place>",
    "attendees": ["email1@example.com", "email2@example.com"],

    // update_event (same fields as create_event, include event_id)
    "event_id": "<id>",

    // delete_event
    "event_id": "<id>"
  },
  "timezone": {
    "iana": "<IANA like Asia/Kolkata>",
    "offset": "+05:30",
    "confidence": 0.0,
    "source": "<settings|calendar_primary|calendar_list>"
  },
  "validation": {
    "timestamps_have_tz": true,
    "within_local_day_boundaries": true
  }
}

Tool Mapping
- Inspect available Composio Calendar tools (function specs).
- Map NIS to tool-specific payloads exactly:
  • list_events → calendar.events.list | calendarList.list
  • free_busy → freeBusy.query
  • get_event → events.get
  • create_event → events.insert
  • update_event → events.patch or events.update
  • delete_event → events.delete
- Respect each tool’s schema (names, nesting). If a field is unsupported, omit it.

Execution Plan (every request)
1) Determine action from the user request.
2) Run Timezone Discovery Protocol. Abort with a single question if not ≥ 0.95 confidence.
3) Build the NIS JSON. Validate timestamps_have_tz = true.
4) Convert NIS → tool payload. Execute via Composio tool.
5) Transform tool response → “User Markdown Output”.
6) Do not include internal JSON or tool traces in the final answer.

User Markdown Output
- List / Summary (events)
  Title: "📅 Upcoming Events ({{ '{{' }} timezone.iana {{ '}}' }} / UTC{{ '{{' }} timezone.offset {{ '}}' }})"
  For each event (one line each):
  - "• {{ '{{' }} summary {{ '}}' }} — {{ '{{' }} local_day_name {{ '}}' }}, {{ '{{' }} local_date {{ '}}' }} {{ '{{' }} start_local {{ '}}' }}–{{ '{{' }} end_local {{ '}}' }} (UTC{{ '{{' }} offset {{ '}}' }}){{ '{{' }} location_opt {{ '}}' }}{{ '{{' }} attendees_opt {{ '}}' }}  {{ '{{' }} link_opt {{ '}}' }}"
- Free/Busy
  Title: "🧭 Availability ({{ '{{' }} range_local {{ '}}' }}, {{ '{{' }} timezone.iana {{ '}}' }})"
  - "🟢 Free: {{ '{{' }} slots_free {{ '}}' }}"
  - "🔴 Busy: {{ '{{' }} slots_busy {{ '}}' }}"
- Create/Update/Delete
  - "✅ {{ '{{' }} action|title {{ '}}' }}: {{ '{{' }} summary {{ '}}' }} — {{ '{{' }} local_date {{ '}}' }} {{ '{{' }} start_local {{ '}}' }}–{{ '{{' }} end_local {{ '}}' }} ({{ '{{' }} timezone.iana {{ '}}' }}, UTC{{ '{{' }} offset {{ '}}' }}) {{ '{{' }} link_opt {{ '}}' }}"
- No results
  - "ℹ️ No matching events."

Guardrails
- Never output the NIS or tool payloads. Only final Markdown.
- Never use naive timestamps (must have +HH:MM or explicit timeZone fields).
- Never use system timezone or locale. Only verified Calendar timezone.
- If timezone cannot be verified, ask the single clarifying question and stop.
- Respect user’s exact intent and constraints.
"""

# Compact prompt (fast mode): minimal instructions to reduce tokens and latency
SYSTEM_PROMPT_COMPACT_JINJA = """
Google Calendar Agent — precise and concise.

Now: {{ current_time }}  |  TZ hint: {{ tz_hint or "unknown" }}  |  Max results: {{ max_results }}

Do:
- Discover timezone via tools (settings/timezone → calendar primary → calendar list). If not ≥0.95 confidence, ask user which timezone to use and stop.
- Build internal NIS {action, params, timezone} (do not show to user).
- Map NIS → Composio tool; execute.
- Output Markdown with emojis only (no JSON/tool traces).

Rules:
- Always include timezone in timestamps (ISO +offset or {dateTime+timeZone}).
- Resolve relative dates in discovered timezone.
- Defaults: create=30m; list window=[now, now+{{ default_window_days }}d]. All-day uses date.

Output:
- List: "📅 Upcoming Events ({{ '{{' }} timezone.iana {{ '}}' }} / UTC{{ '{{' }} timezone.offset {{ '}}' }})" then one bullet per event.
- Free/busy: "🧭 Availability (range, tz)" with 🟢 Free / 🔴 Busy.
- Create/Update/Delete: "✅ <Action>: <title> — <local date> <start–end> (tz, UTC±)" + link.
- No results: "ℹ️ No matching events."
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the optimized Calendar agent (prompt-driven schema + formatting).")
    parser.add_argument("uuid", help="Composio external user_id or stored record_id")
    parser.add_argument("prompt", nargs="?", help="Optional prompt. Defaults to a Calendar summary task.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and latency metrics")
    parser.add_argument("--fast", action="store_true", help="Enable compact prompt and fast tool filtering")
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} in environment")
    return value


def _model_dump(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            pass
    return {"value": str(obj)}


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps({"value": str(obj)})


def _load_all_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _load_record_by_id(path: Path, record_id: str) -> Optional[Dict[str, Any]]:
    for rec in _load_all_records(path):
        if str(rec.get("record_id")) == str(record_id):
            return rec
    return None


def _find_calendar_connected_account_id(records: List[Dict[str, Any]], user_id: str) -> Optional[str]:
    # Prefer explicit google_calendar service
    for rec in records:
        if str(rec.get("user_id")) != str(user_id):
            continue
        if (rec.get("service") or "").lower() == "google_calendar" and rec.get("connected_account_id"):
            return str(rec.get("connected_account_id"))
    # Fallback to any record listing Calendar-like toolkits
    for rec in records:
        if str(rec.get("user_id")) != str(user_id):
            continue
        tks = rec.get("toolkits") or []
        if isinstance(tks, list) and any(str(t).upper() in {"GOOGLECALENDAR", "GOOGLE_CALENDAR", "CALENDAR"} for t in tks):
            if rec.get("connected_account_id"):
                return str(rec.get("connected_account_id"))
    return None


def normalize_function_payload(tool: Any) -> Optional[Dict[str, Any]]:
    payload = _model_dump(tool)
    fn = payload.get("function") if isinstance(payload, dict) else None
    if isinstance(fn, dict) and "name" in fn:
        return {k: v for k, v in fn.items() if k in ("name", "description", "parameters", "strict")}
    if isinstance(payload, dict) and "name" in payload and "parameters" in payload:
        return {k: v for k, v in payload.items() if k in ("name", "description", "parameters", "strict")}
    return None


class CalendarToolLite(BaseTool):
    name: str
    description: str = ""
    slug: str
    user_id: str
    tools_client: Any
    connected_account_id: Optional[str] = None
    debug: bool = False

    def _run(self, tool_input: Any = None, run_manager: Any = None, **kwargs: Any) -> str:
        # Accept dict, json string, or kwargs; pass through
        if isinstance(tool_input, dict):
            arguments: Dict[str, Any] = tool_input
        elif tool_input is not None:
            try:
                arguments = json.loads(str(tool_input))
            except Exception:
                arguments = {"text": str(tool_input)}
        else:
            arguments = dict(kwargs) if kwargs else {}

        arguments.setdefault("user_id", "me")

        t0 = time.monotonic()
        try:
            raw = self.tools_client.execute(
                slug=self.slug,
                arguments=arguments,
                user_id=self.user_id,
                connected_account_id=self.connected_account_id,
            )
        except Exception as exc:
            return _safe_json({"status": "error", "action": self.slug, "error": str(exc)})
        exec_ms = (time.monotonic() - t0) * 1000.0

        raw_obj = _model_dump(raw)
        payload = raw_obj
        for key in ("result", "data", "output"):
            if isinstance(raw_obj, dict) and key in raw_obj and isinstance(raw_obj[key], (dict, list)):
                payload = raw_obj[key]
                break

        if self.debug:
            try:
                size = None
                if isinstance(payload, dict):
                    if isinstance(payload.get("items"), list):
                        size = len(payload.get("items"))
                    elif isinstance(payload.get("events"), list):
                        size = len(payload.get("events"))
                print(f"[tool] {self.slug} exec_ms={exec_ms:.1f} items={size if size is not None else 'N/A'}")
            except Exception:
                pass

        return _safe_json({"status": "success", "action": self.slug, "result": payload})


def build_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    ).partial(system_prompt=system_prompt)


class DebugCallback(BaseCallbackHandler):
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._llm_t0: Optional[float] = None
        self._tool_t0: Optional[float] = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        if not self.enabled:
            return
        self._llm_t0 = time.monotonic()
        try:
            preview = " | ".join(p[:120].replace("\n", " ") for p in prompts or [])
            print(f"[llm] start prompts={len(prompts or [])} preview={preview}")
        except Exception:
            print("[llm] start")

    def on_llm_end(self, response, **kwargs):
        if not self.enabled:
            return
        ms = (time.monotonic() - (self._llm_t0 or time.monotonic())) * 1000.0
        print(f"[llm] end latency_ms={ms:.1f}")

    def on_agent_action(self, action, **kwargs):
        if not self.enabled:
            return
        try:
            log = getattr(action, "log", "")
            first = (log or "").splitlines()[:3]
            tool = getattr(action, "tool", "?")
            print(f"[reason] {' / '.join(first)}")
            print(f"[act] tool={tool} input_preview={str(action.tool_input)[:160]}")
        except Exception:
            pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        if not self.enabled:
            return
        self._tool_t0 = time.monotonic()
        name = (serialized or {}).get("name") or "<tool>"
        print(f"[tool] start {name} args_preview={str(input_str)[:160]}")

    def on_tool_end(self, output, **kwargs):
        if not self.enabled:
            return
        ms = (time.monotonic() - (self._tool_t0 or time.monotonic())) * 1000.0
        preview = str(output)[:200].replace("\n", " ")
        print(f"[tool] end latency_ms={ms:.1f} output_preview={preview}")

    def on_agent_finish(self, finish, **kwargs):
        if not self.enabled:
            return
        try:
            log = getattr(finish, "log", "")
            last = (log or "").splitlines()[-1] if log else ""
            print(f"[final] {last[:200]}")
        except Exception:
            pass

def main() -> None:
    args = parse_args()
    debug = args.debug or os.getenv("CAL_AGENT_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    fast = args.fast or os.getenv("CAL_AGENT_FAST", "").lower() in {"1", "true", "yes", "on"}
    t_start = time.monotonic()

    composio_api_key = require_env("COMPOSIO_API_KEY")
    openai_api_key = require_env("OPENAI_API_KEY")

    cli_id = args.uuid
    external_user_id = cli_id
    connected_account_id: Optional[str] = None
    rec = _load_record_by_id(DEFAULT_RECORD_PATH, cli_id)
    if rec:
        external_user_id = str(rec.get("user_id") or external_user_id)
        all_recs = _load_all_records(DEFAULT_RECORD_PATH)
        connected_account_id = _find_calendar_connected_account_id(all_recs, external_user_id)
    user_prompt = args.prompt or f"Summarize my calendar for the next {DEFAULT_WINDOW_DAYS} days."

    composio = Composio(api_key=composio_api_key)
    chat_model = ChatOpenAI(model=args.model, openai_api_key=openai_api_key, temperature=0)

    # Build Jinja system prompt
    if debug:
        print("[step] Building system prompt (Jinja)")
    now_utc_iso = (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    tz_hint = None
    try:
        # If previous calendar record stored a timezone, pass as hint only
        for rec_all in _load_all_records(DEFAULT_RECORD_PATH):
            if str(rec_all.get("user_id")) == str(external_user_id) and rec_all.get("service") == "google_calendar":
                if isinstance(rec_all.get("timezone"), str) and rec_all.get("timezone"):
                    tz_hint = rec_all.get("timezone")
                    break
    except Exception:
        tz_hint = None

    context = {
        "current_time": now_utc_iso,
        "tz_hint": tz_hint or "",
        "max_results": CALENDAR_MAX_RESULTS,
        "default_window_days": DEFAULT_WINDOW_DAYS,
        "user_id": external_user_id,
        "connected_account_id": connected_account_id or "",
    }
    tpl = SYSTEM_PROMPT_COMPACT_JINJA if fast else SYSTEM_PROMPT_JINJA
    system_text = jinja2_formatter(tpl, **context)
    prompt_template = build_prompt_template(system_text)

    # Fetch Calendar tools via cache or first working toolkit
    if debug:
        print("[step] Fetching tools")
    t_fetch = time.monotonic()
    cached_functions = _load_cached_functions(external_user_id)
    functions: List[Dict[str, Any]] = []
    lc_tools: List[BaseTool] = []
    if cached_functions:
        if debug:
            print("[cache] Using cached tool functions")
        functions = cached_functions
    else:
        for slug in GOOGLE_CALENDAR_TOOLKIT_CANDIDATES:
            try:
                got = composio.tools.get(user_id=external_user_id, toolkits=[slug])
            except Exception:
                got = []
            seen_names: set[str] = set()
            tmp_functions: List[Dict[str, Any]] = []
            for t in got:
                fn = normalize_function_payload(t)
                if not fn:
                    continue
                name = str(fn.get("name", ""))
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                tmp_functions.append(fn)
            if tmp_functions:
                functions = tmp_functions
                break
        if not functions:
            print(f"No Calendar tools available for user {external_user_id}. Connect Calendar and retry.")
            return
        _save_cached_functions(external_user_id, functions)

    for fn in functions:
        lc_tools.append(
            CalendarToolLite(
                name=str(fn.get("name")),
                description=str(fn.get("description") or ""),
                slug=str(fn.get("name")),
                user_id=external_user_id,
                tools_client=composio.tools,
                connected_account_id=connected_account_id,
                debug=debug,
            )
        )
    tools_ms = (time.monotonic() - t_fetch) * 1000.0

    # Optional fast filtering: limit tools to a likely subset based on prompt
    if fast:
        p = (user_prompt or "").lower()
        def intent() -> str:
            if any(k in p for k in ["free", "availability", "busy", "free/busy", "slot"]):
                return "FREEBUSY"
            if any(k in p for k in ["create", "add", "schedule", "make an event", "book"]):
                return "CREATE"
            if any(k in p for k in ["update", "reschedule", "move", "change"]):
                return "UPDATE"
            if any(k in p for k in ["delete", "cancel", "remove"]):
                return "DELETE"
            if any(k in p for k in ["list", "show", "what's on", "events", "this week", "today", "tomorrow"]):
                return "LIST"
            if any(k in p for k in ["get event", "event id", "details"]):
                return "GET"
            return "OTHER"
        goal = intent()
        def keep_fn(fn: Dict[str, Any]) -> bool:
            n = str(fn.get("name", "")).upper()
            if goal == "FREEBUSY":
                return ("FREE" in n and "BUSY" in n) or "FREEBUSY" in n
            if goal == "CREATE":
                return any(k in n for k in ["INSERT", "CREATE", "ADD"]) and "EVENT" in n
            if goal == "UPDATE":
                return any(k in n for k in ["PATCH", "UPDATE"]) and "EVENT" in n
            if goal == "DELETE":
                return "DELETE" in n and "EVENT" in n
            if goal == "LIST":
                return any(k in n for k in ["LIST", "SEARCH"]) and "EVENT" in n
            if goal == "GET":
                return ("GET" in n or "GET_EVENT" in n) and "EVENT" in n
            return True
        if debug:
            print(f"[fast] intent={goal}")
        filtered_functions = [f for f in functions if keep_fn(f)]
        if filtered_functions:
            # Rebuild lc_tools to match filtered functions
            functions = filtered_functions
            lc_tools = [
                CalendarToolLite(
                    name=str(fn.get("name")),
                    description=str(fn.get("description") or ""),
                    slug=str(fn.get("name")),
                    user_id=external_user_id,
                    tools_client=composio.tools,
                    connected_account_id=connected_account_id,
                    debug=debug,
                ) for fn in functions
            ]

    print("[✓] Requested toolkits: GOOGLE_CALENDAR")
    print(f"[i] Using user_id: {external_user_id}")
    if connected_account_id:
        print(f"[i] Calendar connected_account_id: {connected_account_id}")
    print(f"[i] Exposed Calendar tools: {len(lc_tools)}")
    if args.prompt is None:
        print(f"[i] Using fallback prompt: {user_prompt}")
    if debug:
        print(f"[latency] tools_fetch_ms={tools_ms:.1f}")

    if debug:
        print("[step] Creating agent executor")
    agent = create_openai_functions_agent(chat_model, functions, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=lc_tools, verbose=False)

    print(f"Task: {user_prompt}")
    if debug:
        print("[step] Invoking agent")
    t_agent = time.monotonic()
    dbg = DebugCallback(enabled=debug)
    result = agent_executor.invoke({"input": user_prompt}, config={"callbacks": [dbg]})
    agent_ms = (time.monotonic() - t_agent) * 1000.0
    output = result.get("output") if isinstance(result, dict) else result
    print(output)
    if debug:
        total_ms = (time.monotonic() - t_start) * 1000.0
        print(f"[latency] agent_invoke_ms={agent_ms:.1f} total_ms={total_ms:.1f}")


if __name__ == "__main__":
    main()
