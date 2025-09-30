import argparse
import base64
import html
import json
import os
import sys
import threading
import traceback
import time
import quopri
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

def _install_thread_excepthook():
    def _hook(args: threading.ExceptHookArgs):
        try:
            tb = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        except Exception:
            tb = ""
        name = getattr(args.thread, "name", "")
        if "_thread_loop" in name and "composio/core/models/_telemetry" in tb:
            return
        sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)

    threading.excepthook = _hook

_install_thread_excepthook()

from composio import Composio

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler


# Ensure a writable cache path for Composio
os.environ.setdefault("COMPOSIO_CACHE_DIR", str((Path.cwd() / ".composio_cache").resolve()))

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_RECORD_PATH = Path(os.getenv("ACCOUNT_OUTPUT_PATH", "outbox/connected_accounts.jsonl"))
EMAIL_SUMMARY_DAYS = int(os.getenv("EMAIL_SUMMARY_DAYS", "7"))
EMAIL_MAX_RESULTS = int(os.getenv("EMAIL_MAX_RESULTS", "25"))
EMAIL_BODY_MAX_CHARS = int(os.getenv("EMAIL_BODY_MAX_CHARS", "500"))
FALLBACK_PROMPT = os.getenv(
    "FALLBACK_TASK",
    f"Summarize my emails from the last {EMAIL_SUMMARY_DAYS} days.",
)


SYSTEM_PROMPT_TEMPLATE = """
You are a Gmail Orchestrator Agent with access to Composio MCP tools for Gmail only.
Interpret user instructions, choose the correct Gmail tool, and produce a concise outcome.

## Identity
- The user is identified by a UUID provided at runtime.
- Act only on behalf of this user's connected Gmail account; use user_id='me' for Gmail API calls.

## Rules
1. Use only the exposed Gmail tools (names start with GMAIL_). Do not fabricate results.
2. For search/summarize tasks, build a Gmail query as needed and request at most %%EMAIL_MAX_RESULTS%% results (set max_results accordingly). Exclude drafts unless the user requests drafts.
   - If nextPageToken is returned, continue until the requested count is met or the token is absent.
3. When retrieving messages, you will receive only subject and body (plain text). The body is already decoded and trimmed by the runtime.
4. Summaries must be a point-wise list: one bullet per email (up to %%EMAIL_MAX_RESULTS%%). Each bullet should start with the subject, then a short gist from the body.
   - Keep each bullet to one line when possible.
5. When asked to send an email, call the appropriate Gmail send tool and provide all required parameters from the user prompt (do not invent recipients).
6. Output only the user-facing result (no JSON or tool traces).
7. If there are no recent emails, reply: 'No recent emails to summarize.'

## Generalized Input Schema (Guidance)
- Always follow the exact JSON schema of the selected GMAIL_* tool.
- Use these conventions when mapping natural language to tool arguments (only if the tool schema matches):
  • Send (e.g., GMAIL_SEND_*): {recipient_email: string, subject: string, body: string, cc?: string[], bcc?: string[], is_html?: boolean}. Do not fabricate recipients.
  • Search/List (e.g., GMAIL_FETCH_*/GMAIL_SEARCH_*): {query: string, max_results?: number ≤ %%EMAIL_MAX_RESULTS%%}. Prefer queries like newer_than:%%EMAIL_SUMMARY_DAYS%%d and exclude drafts unless asked.
  • Read (e.g., GMAIL_GET_MESSAGE/GMAIL_GET_THREAD): {id: string}.
- The runtime automatically supplies authentication context (user_id). Do not include it.
- If the tool exposes cc/bcc arrays, pass arrays of strings (not comma-joined).
- If the body contains HTML and the tool supports it, set is_html = true.
"""

SYSTEM_PROMPT = (
    SYSTEM_PROMPT_TEMPLATE
    .replace("%%EMAIL_MAX_RESULTS%%", str(EMAIL_MAX_RESULTS))
    .replace("%%EMAIL_SUMMARY_DAYS%%", str(EMAIL_SUMMARY_DAYS))
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Gmail-only LangChain + Composio agent.")
    parser.add_argument("uuid", help="Composio external user_id or stored record_id")
    parser.add_argument("prompt", nargs="?", help="Optional prompt. Defaults to a Gmail summary task.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL}).")
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} in environment")
    return value


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


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps({"value": str(obj)})


def _load_all_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    recs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def _load_record_by_id(path: Path, record_id: str) -> Optional[Dict[str, Any]]:
    for rec in _load_all_records(path):
        if str(rec.get("record_id")) == str(record_id):
            return rec
    return None


def _find_gmail_connected_account_id(records: List[Dict[str, Any]], user_id: str) -> Optional[str]:
    # Prefer explicit gmail service
    for rec in records:
        if str(rec.get("user_id")) != str(user_id):
            continue
        svc = (rec.get("service") or "").lower()
        if svc == "gmail" and rec.get("connected_account_id"):
            return str(rec.get("connected_account_id"))
    # Fall back to any record listing GMAIL in toolkits
    for rec in records:
        if str(rec.get("user_id")) != str(user_id):
            continue
        tks = rec.get("toolkits") or []
        if isinstance(tks, list) and any(str(t).upper() == "GMAIL" for t in tks):
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


# -------- Gmail MIME / HTML normalization helpers -------- #


def _b64url_decode(data_str: str) -> bytes:
    try:
        pad = (-len(data_str)) % 4
        data_str = data_str + ("=" * pad)
        return base64.urlsafe_b64decode(data_str.encode("utf-8"))
    except Exception:
        return b""


def _is_quoted_printable(headers: Any) -> bool:
    if not isinstance(headers, list):
        return False
    for h in headers:
        if not isinstance(h, dict):
            continue
        name = str(h.get("name", "")).lower()
        value = str(h.get("value", "")).lower()
        if name == "content-transfer-encoding" and "quoted-printable" in value:
            return True
    return False


def _html_to_text(html_str: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\\1>", " ", html_str)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    try:
        text = html.unescape(text)
    except Exception:
        pass
    text = re.sub(r"\s+", " ", text).strip()
    return text


## Removed unused _parse_iso_date and datetime imports to reduce bloat


def _extract_text_from_payload(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    parts = payload.get("parts")
    headers = payload.get("headers")
    candidates: List[Dict[str, Any]] = []
    if isinstance(parts, list) and parts:
        for p in parts:
            if not isinstance(p, dict):
                continue
            mime = str(p.get("mimeType", "")).lower()
            if mime in {"text/plain", "text/html"}:
                candidates.append(p)
            sub = p.get("parts")
            if isinstance(sub, list):
                for sp in sub:
                    if not isinstance(sp, dict):
                        continue
                    mime2 = str(sp.get("mimeType", "")).lower()
                    if mime2 in {"text/plain", "text/html"}:
                        candidates.append(sp)
    else:
        body = payload.get("body")
        if isinstance(body, dict) and isinstance(body.get("data"), str):
            candidates.append(payload)

    def rank(item: Dict[str, Any]) -> int:
        return 0 if str(item.get("mimeType", "")).lower() == "text/plain" else 1

    candidates.sort(key=rank)
    for part in candidates:
        body = part.get("body") if isinstance(part, dict) else None
        data_str = body.get("data") if isinstance(body, dict) else None
        if isinstance(data_str, str):
            raw = _b64url_decode(data_str)
            if not raw:
                continue
            # Decode quoted-printable if indicated on the part or payload headers
            try:
                part_headers = []
                if isinstance(part, dict) and isinstance(part.get("headers"), list):
                    part_headers = part.get("headers")
                qp_headers = part_headers or headers or []
                if _is_quoted_printable(qp_headers):
                    raw = quopri.decodestring(raw)
            except Exception:
                pass
            try:
                txt = raw.decode("utf-8", errors="replace")
            except Exception:
                txt = raw.decode(errors="replace")
            mime = str(part.get("mimeType", "")).lower()
            if mime == "text/html":
                txt = _html_to_text(txt)
            return txt.strip()
    return None


def _extract_text_from_parts(parts: Any, headers: Any = None) -> Optional[str]:
    if not isinstance(parts, list):
        return None
    # Reuse payload logic by wrapping parts as payloads
    payload_like = {"parts": parts, "headers": headers or []}
    return _extract_text_from_payload(payload_like)


def _header(headers: List[Dict[str, Any]], name: str) -> Optional[str]:
    for h in headers or []:
        if isinstance(h, dict) and h.get("name", "").lower() == name.lower():
            return str(h.get("value"))
    return None


def _extract_text_from_any_message(msg: Dict[str, Any]) -> Optional[str]:
    # Prefer preview body if available
    preview = msg.get("preview")
    if isinstance(preview, dict) and isinstance(preview.get("body"), str) and preview.get("body").strip():
        return preview.get("body").strip()
    # Top-level parts
    if isinstance(msg.get("parts"), list):
        txt = _extract_text_from_parts(msg.get("parts"), msg.get("headers"))
        if txt:
            return txt
    # Payload tree
    payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else None
    if payload:
        txt = _extract_text_from_payload(payload)
        if txt:
            return txt
    # Snippet fallback
    sn = msg.get("snippet")
    if isinstance(sn, str) and sn.strip():
        return sn.strip()
    return None


def normalize_gmail_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    # Only subject and body as requested
    payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else None
    headers = msg.get("headers") if isinstance(msg.get("headers"), list) else []
    if not headers and isinstance(payload, dict):
        headers = payload.get("headers") if isinstance(payload.get("headers"), list) else []

    subject = _header(headers, "Subject") or msg.get("subject") or (msg.get("preview") or {}).get("subject")
    body_text = _extract_text_from_any_message(msg)
    if body_text and len(body_text) > EMAIL_BODY_MAX_CHARS:
        body_text = body_text[: EMAIL_BODY_MAX_CHARS] + "…"

    return {
        "subject": subject,
        "body": body_text,
    }


def normalize_gmail_output(data: Dict[str, Any]) -> Any:
    # List responses → only subject/body for each message
    if isinstance(data.get("messages"), list) or isinstance(data.get("emails"), list):
        items = data.get("messages") or data.get("emails") or []
        out_list: List[Dict[str, Any]] = []
        for item in items[: EMAIL_MAX_RESULTS]:
            if isinstance(item, dict):
                out_list.append(normalize_gmail_message(item))
        return {"messages": out_list}
    # Single message shape
    if data.get("id") and (data.get("payload") or data.get("snippet") or data.get("parts")):
        return normalize_gmail_message(data)
    # Generic
    return data


class GmailTool(BaseTool):
    name: str
    description: str = ""
    slug: str
    user_id: str
    tools_client: Any
    connected_account_id: Optional[str] = None

    def _run(self, tool_input: Any = None, run_manager: Any = None, **kwargs: Any) -> str:
        # Accept dict, string (json or text), or kwargs
        if isinstance(tool_input, dict):
            arguments: Dict[str, Any] = tool_input
        elif tool_input is not None:
            try:
                arguments = json.loads(str(tool_input))
            except Exception:
                arguments = {"text": str(tool_input)}
        else:
            arguments = dict(kwargs) if kwargs else {}

        slug_up = self.slug.upper()

        # Common defaults
        arguments.setdefault("user_id", "me")

        # No parameter normalization here; rely on tool schemas and prompt

        try:
            raw = self.tools_client.execute(
                slug=self.slug,
                arguments=arguments,
                user_id=self.user_id,
                connected_account_id=self.connected_account_id,
            )
        except Exception as exc:
            return _safe_json({"status": "error", "action": self.slug, "error": str(exc)})

        # Unwrap common Composio shapes (e.g., {result: {...}, successful: true, error: null})
        raw_obj = _model_dump(raw)
        payload = raw_obj
        for key in ("result", "data", "output"):
            if isinstance(raw_obj, dict) and key in raw_obj and isinstance(raw_obj[key], (dict, list)):
                payload = raw_obj[key]
                break

        normalized = normalize_gmail_output(payload)
        return _safe_json({"status": "success", "action": self.slug, "result": normalized})


class StepLogger(BaseCallbackHandler):
    """Logs LLM and tool steps with latencies by default (no --debug flag).

    Safe to attach via callbacks=[StepLogger()] on chain/agent invocations.
    """

    def __init__(self) -> None:
        self._tool_start: dict[object, float] = {}
        self._llm_start: dict[object, float] = {}
        self.tool_total: float = 0.0
        self.llm_total: float = 0.0
        self.tool_calls: int = 0

    # LLM timings
    def on_llm_start(self, serialized, prompts, **kwargs):  # type: ignore[no-untyped-def]
        run_id = kwargs.get("run_id") or id(prompts)
        self._llm_start[run_id] = time.perf_counter()
        try:
            n = len(prompts) if isinstance(prompts, (list, tuple)) else 1
        except Exception:
            n = 1
        print(f"[llm] start ({n} prompt{'s' if n != 1 else ''})")

    def on_llm_end(self, response, **kwargs):  # type: ignore[no-untyped-def]
        run_id = kwargs.get("run_id") or id(response)
        t0 = self._llm_start.pop(run_id, None)
        if t0 is not None:
            dt = time.perf_counter() - t0
            self.llm_total += dt
            print(f"[llm] done in {dt:.2f}s")

    # Tool timings
    def on_tool_start(self, serialized, input_str, **kwargs):  # type: ignore[no-untyped-def]
        run_id = kwargs.get("run_id") or id(input_str)
        self._tool_start[run_id] = time.perf_counter()
        name = None
        if isinstance(serialized, dict):
            name = serialized.get("name")
        name = name or "<tool>"
        try:
            inp = input_str if isinstance(input_str, str) else json.dumps(input_str)
        except Exception:
            inp = str(input_str)
        if isinstance(inp, str) and len(inp) > 500:
            inp = inp[:500] + "…"
        print(f"[tool] {name} -> {inp}")

    def on_tool_end(self, output, **kwargs):  # type: ignore[no-untyped-def]
        run_id = kwargs.get("run_id") or id(output)
        t0 = self._tool_start.pop(run_id, None)
        try:
            out = output if isinstance(output, str) else json.dumps(output)
        except Exception:
            out = str(output)
        if isinstance(out, str) and len(out) > 300:
            out = out[:300] + "…"
        if t0 is not None:
            dt = time.perf_counter() - t0
            self.tool_total += dt
            self.tool_calls += 1
            print(f"[tool] done in {dt:.2f}s -> {out}")
        else:
            print(f"[tool] done -> {out}")


def build_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    return prompt.partial(system_prompt=system_prompt)


# Intents are inferred by the LLM using the system prompt and tool schemas


def main() -> None:
    args = parse_args()

    composio_api_key = require_env("COMPOSIO_API_KEY")
    openai_api_key = require_env("OPENAI_API_KEY")

    cli_id = args.uuid
    # Resolve identity: accept record_id or user_id
    external_user_id = cli_id
    connected_account_id: Optional[str] = None
    rec = _load_record_by_id(DEFAULT_RECORD_PATH, cli_id)
    if rec:
        external_user_id = str(rec.get("user_id") or external_user_id)
        all_recs = _load_all_records(DEFAULT_RECORD_PATH)
        connected_account_id = _find_gmail_connected_account_id(all_recs, external_user_id)
    user_prompt = args.prompt or FALLBACK_PROMPT

    composio = Composio(api_key=composio_api_key)
    chat_model = ChatOpenAI(model=args.model, openai_api_key=openai_api_key, temperature=0)
    prompt_template = build_prompt_template(SYSTEM_PROMPT)

    requested_toolkits = ["GMAIL"]
    raw_tools = composio.tools.get(user_id=external_user_id, toolkits=requested_toolkits)
    if not raw_tools:
        print(f"No Gmail tools available for user {external_user_id}. Connect Gmail and retry.")
        return

    functions: List[Dict[str, Any]] = []
    lc_tools: List[BaseTool] = []
    for t in raw_tools:
        fn = normalize_function_payload(t)
        if not fn:
            continue
        name_up = str(fn.get("name", "")).upper()
        if not name_up.startswith("GMAIL_"):
            continue
        functions.append(fn)
        lc_tools.append(
            GmailTool(
                name=str(fn.get("name")),
                description=str(fn.get("description") or ""),
                slug=str(fn.get("name")),
                user_id=external_user_id,
                tools_client=composio.tools,
                connected_account_id=connected_account_id,
            )
        )

    print("[✓] Requested toolkits: GMAIL")
    print(f"[i] Using user_id: {external_user_id}")
    if connected_account_id:
        print(f"[i] Gmail connected_account_id: {connected_account_id}")
    print(f"[i] Exposed Gmail tools: {len(lc_tools)}")
    if args.prompt is None:
        print(f"[i] Using fallback prompt: {user_prompt}")

    # No intent gating; expose all Gmail tools
    if not functions or not lc_tools:
        print("No Gmail tools available after discovery. Adjust your prompt or connection.")
        return

    agent = create_openai_functions_agent(chat_model, functions, prompt_template)
    # Reduce token and console bloat during tool runs
    agent_executor = AgentExecutor(agent=agent, tools=lc_tools, verbose=False)

    print(f"Task: {user_prompt}")
    result = agent_executor.invoke({"input": user_prompt})
    output = result.get("output") if isinstance(result, dict) else result
    print(output)


if __name__ == "__main__":
    main()
