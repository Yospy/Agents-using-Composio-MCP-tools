import argparse
import json
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Suppress Composio telemetry thread tracebacks
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

# Ensure writable cache for Composio
os.environ.setdefault("COMPOSIO_CACHE_DIR", str((Path.cwd() / ".composio_cache").resolve()))

load_dotenv()

# Configuration
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_RECORD_PATH = Path(os.getenv("ACCOUNT_OUTPUT_PATH", "outbox/connected_accounts.jsonl"))
FALLBACK_PROMPT = "List my Notion pages and databases."
NOTION_TOOLKIT_CANDIDATES = ["NOTION", "NOTIONHQ", "NOTION_DB", "NOTIONDATABASES"]

SYSTEM_PROMPT = """You are a Notion agent with access to Composio MCP tools for Notion.

## Your Role
Help users interact with their Notion workspace using the available NOTION_* tools.

## Guidelines
1. Use only the exposed Notion tools (names start with NOTION_)
2. For listing operations, provide clear summaries
3. For create/update operations, confirm what was done
4. If a tool fails, explain the error clearly
5. Be concise and helpful

## Common Operations
- List pages/databases: Use list or search tools
- Create pages: Use create tools with title and content
- Update pages: Use update tools with page_id
- Search: Use search tools with query parameters

Output clean, user-friendly responses."""


class NotionTool(BaseTool):
    """Minimal wrapper for Composio Notion tools."""
    name: str
    description: str = ""
    slug: str
    user_id: str
    tools_client: Any
    connected_account_id: Optional[str] = None

    def _run(self, tool_input: Any = None, run_manager: Any = None, **kwargs: Any) -> str:
        # Parse input - accept dict, JSON string, or kwargs
        if isinstance(tool_input, dict):
            arguments = tool_input
        elif tool_input is not None:
            try:
                arguments = json.loads(str(tool_input))
            except Exception:
                arguments = {"text": str(tool_input)}
        else:
            arguments = dict(kwargs) if kwargs else {}

        # Execute tool via Composio
        try:
            raw = self.tools_client.execute(
                slug=self.slug,
                arguments=arguments,
                user_id=self.user_id,
                connected_account_id=self.connected_account_id,
            )
        except Exception as exc:
            return json.dumps({"status": "error", "action": self.slug, "error": str(exc)})

        # Convert response to dict
        if isinstance(raw, dict):
            result = raw
        elif hasattr(raw, "model_dump"):
            result = raw.model_dump()
        elif hasattr(raw, "dict"):
            result = raw.dict()
        elif hasattr(raw, "__dict__"):
            result = dict(raw.__dict__)
        else:
            result = {"value": str(raw)}

        return json.dumps({"status": "success", "action": self.slug, "result": result})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Notion LangChain agent with verbose logging.")
    parser.add_argument("uuid", help="Composio external user_id or stored record_id")
    parser.add_argument("prompt", nargs="?", help="Optional task prompt")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
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


def _find_notion_connected_account_id(records: List[Dict[str, Any]], user_id: str) -> Optional[str]:
    # Prefer explicit notion service
    for rec in records:
        if str(rec.get("user_id")) != str(user_id):
            continue
        svc = (rec.get("service") or "").lower()
        if svc == "notion" and rec.get("connected_account_id"):
            return str(rec.get("connected_account_id"))
    # Fall back to any record listing NOTION in toolkits
    for rec in records:
        if str(rec.get("user_id")) != str(user_id):
            continue
        tks = rec.get("toolkits") or []
        if isinstance(tks, list) and any(str(t).upper() in {"NOTION", "NOTIONHQ", "NOTION_DB", "NOTIONDATABASES"} for t in tks):
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


def main() -> None:
    args = parse_args()

    # Load API keys
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
        connected_account_id = _find_notion_connected_account_id(all_recs, external_user_id)
    user_prompt = args.prompt or FALLBACK_PROMPT

    # Initialize clients
    composio = Composio(api_key=composio_api_key)
    chat_model = ChatOpenAI(model=args.model, openai_api_key=openai_api_key, temperature=0)

    # Try to get Notion tools with multiple toolkit candidates
    print("[i] Fetching Notion tools...")
    raw_tools = []
    used_toolkit = None
    for toolkit in NOTION_TOOLKIT_CANDIDATES:
        try:
            raw_tools = composio.tools.get(user_id=external_user_id, toolkits=[toolkit])
            if raw_tools:
                used_toolkit = toolkit
                break
        except Exception:
            continue

    if not raw_tools:
        print(f"[!] No Notion tools found for user {external_user_id}. Verify connection and toolkit permissions.")
        return

    print(f"[✓] Found {len(raw_tools)} tools via toolkit '{used_toolkit}'")

    # Build function schemas and tool wrappers
    functions: List[Dict[str, Any]] = []
    lc_tools: List[BaseTool] = []

    for tool in raw_tools:
        fn = normalize_function_payload(tool)
        if not fn:
            continue
        name_up = str(fn.get("name", "")).upper()
        if not name_up.startswith("NOTION"):
            continue
        functions.append(fn)
        lc_tools.append(
            NotionTool(
                name=str(fn.get("name")),
                description=str(fn.get("description") or ""),
                slug=str(fn.get("name")),
                user_id=external_user_id,
                tools_client=composio.tools,
                connected_account_id=connected_account_id,
            )
        )

    print("[✓] Requested toolkits: " + ", ".join(NOTION_TOOLKIT_CANDIDATES))
    print(f"[i] Using user_id: {external_user_id}")
    if connected_account_id:
        print(f"[i] Notion connected_account_id: {connected_account_id}")
    print(f"[i] Exposed Notion tools: {len(lc_tools)}")
    if args.prompt is None:
        print(f"[i] Using fallback prompt: {user_prompt}")

    if not lc_tools:
        print("[!] No valid Notion tools after filtering.")
        return

    print(f"[i] Task: {user_prompt}\n")

    # Build prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Create agent with verbose logging enabled
    agent = create_openai_functions_agent(chat_model, functions, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=lc_tools,
        verbose=True,  # Shows all agent thinking!
    )

    # Execute
    result = agent_executor.invoke({"input": user_prompt})
    output = result.get("output") if isinstance(result, dict) else result

    print("\n" + "="*50)
    print("FINAL OUTPUT:")
    print("="*50)
    print(output)


if __name__ == "__main__":
    main()