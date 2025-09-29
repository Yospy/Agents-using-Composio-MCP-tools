import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from composio import Composio

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool


# Ensure a writable cache path for Composio
os.environ.setdefault("COMPOSIO_CACHE_DIR", str((Path.cwd() / ".composio_cache").resolve()))

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FALLBACK_PROMPT = os.getenv(
    "FALLBACK_TASK",
    "Create a Google Doc titled 'Weekly Summary' with a short outline of this week's work.",
)
DEFAULT_RECORD_PATH = Path(os.getenv("ACCOUNT_OUTPUT_PATH", "outbox/connected_accounts.jsonl"))


SYSTEM_PROMPT = (
    "You are a Google Docs Orchestrator Agent with access to Composio MCP tools for Google Docs only.\n"
    "Interpret user instructions, select the correct Docs tool, and return clean JSON.\n\n"
    "## Identity\n"
    "- The user is identified by a UUID provided at runtime.\n"
    "- Only act on behalf of this user's connected Google Docs/Drive account.\n\n"
    "## Rules\n"
    "1. Always prefer structured Docs tool calls (GOOGLEDOCS_*).\n"
    "2. Never dump raw payloads; normalize responses.\n"
    "3. Document results must include documentId, title, and link when available.\n"
    "4. If a request contains a Google Docs URL, extract documentId from it.\n"
    "5. If only a title is given, use Drive search (GOOGLEDRIVE_* search/list tools) to find the documentId by exact title before reading/summarizing.\n"
    "6. If a required ID or title is missing and cannot be inferred, ask for it succinctly.\n"
    "7. If a tool fails, retry once with smaller inputs (e.g., shorter content).\n"
    "8. Always respond in JSON with a steps array of actions and normalized results.\n\n"
    "## Example Tool Info (JSON schemas)\n"
    "GOOGLEDOCS_CREATE_DOCUMENT:\n"
    "{\n"
    "  \"name\": \"GOOGLEDOCS_CREATE_DOCUMENT\",\n"
    "  \"description\": \"Create a new Google Doc\",\n"
    "  \"parameters\": {\n"
    "    \"title\": { \"type\": \"string\", \"description\": \"Document title\" },\n"
    "    \"body\": { \"type\": \"string\", \"description\": \"Initial content (optional)\" }\n"
    "  }\n"
    "}\n\n"
    "GOOGLEDOCS_GET_DOCUMENT_BY_ID:\n"
    "{\n"
    "  \"name\": \"GOOGLEDOCS_GET_DOCUMENT_BY_ID\",\n"
    "  \"description\": \"Fetch a document by its ID\",\n"
    "  \"parameters\": {\n"
    "    \"document_id\": { \"type\": \"string\", \"description\": \"Docs document ID\" }\n"
    "  }\n"
    "}\n\n"
    "GOOGLEDOCS_APPEND_PARAGRAPH:\n"
    "{\n"
    "  \"name\": \"GOOGLEDOCS_APPEND_PARAGRAPH\",\n"
    "  \"description\": \"Append text as a paragraph to a document\",\n"
    "  \"parameters\": {\n"
    "    \"document_id\": { \"type\": \"string\" },\n"
    "    \"content\": { \"type\": \"string\" }\n"
    "  }\n"
    "}\n\n"
    "## Output Format\n"
    "{\n"
    "  \"status\": \"success\",\n"
    "  \"steps\": [\n"
    "    {\"action\": \"GOOGLEDOCS_CREATE_DOCUMENT\", \"result\": {...}},\n"
    "    {\"action\": \"GOOGLEDOCS_APPEND_PARAGRAPH\", \"result\": {...}}\n"
    "  ]\n"
    "}\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Docs-only LangChain + Composio agent.")
    parser.add_argument("uuid", help="External user UUID (Composio identity)")
    parser.add_argument("prompt", nargs="?", help="Optional prompt. Defaults to a Docs task.")
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


def normalize_function_payload(tool: Any) -> Optional[Dict[str, Any]]:
    payload = _model_dump(tool)
    fn = payload.get("function") if isinstance(payload, dict) else None
    if isinstance(fn, dict) and "name" in fn:
        return {k: v for k, v in fn.items() if k in ("name", "description", "parameters", "strict")}
    if isinstance(payload, dict) and "name" in payload and "parameters" in payload:
        return {k: v for k, v in payload.items() if k in ("name", "description", "parameters", "strict")}
    return None


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


def _find_docs_connected_account_id(records: List[Dict[str, Any]], user_id: str) -> Optional[str]:
    # Prefer explicit google_docs service
    for rec in records:
        if str(rec.get("user_id")) != str(user_id):
            continue
        svc = (rec.get("service") or "").lower()
        if svc in {"google_docs", "googledocs", "docs"} and rec.get("connected_account_id"):
            return str(rec.get("connected_account_id"))
    # Fall back to any record listing GOOGLEDOCS in toolkits
    for rec in records:
        if str(rec.get("user_id")) != str(user_id):
            continue
        tks = rec.get("toolkits") or []
        if isinstance(tks, list) and any(str(t).upper() == "GOOGLEDOCS" for t in tks):
            if rec.get("connected_account_id"):
                return str(rec.get("connected_account_id"))
    return None


def normalize_docs_output(data: Dict[str, Any]) -> Any:
    doc_id = (
        data.get("documentId")
        or data.get("id")
        or (data.get("document") or {}).get("documentId")
    )
    title = data.get("title") or (data.get("document") or {}).get("title")
    link = data.get("link") or data.get("url")
    if doc_id or title or link:
        result = {"documentId": doc_id, "title": title, "link": link}
        # Optional enrichments
        for k in ("word_count", "revision_id", "created_time", "modified_time"):
            if data.get(k) is not None:
                result[k] = data.get(k)
        return result
    # Handle list responses
    if isinstance(data.get("documents"), list):
        out: List[Dict[str, Any]] = []
        for d in data["documents"]:
            if isinstance(d, dict):
                out.append(
                    {
                        "documentId": d.get("documentId") or d.get("id"),
                        "title": d.get("title"),
                        "link": d.get("link") or d.get("url"),
                    }
                )
        return out
    return data


class DocsTool(BaseTool):
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

        # Light arg normalization for common slugs
        slug_up = self.slug.upper()
        # Ensure expected 'id' is present for common read operations
        if ("GET_DOCUMENT" in slug_up) or slug_up.endswith("_GET_DOCUMENT_BY_ID"):
            if "id" not in arguments:
                if "document_id" in arguments:
                    arguments["id"] = arguments.get("document_id")
                elif "documentId" in arguments:
                    arguments["id"] = arguments.get("documentId")

        try:
            raw = self.tools_client.execute(
                slug=self.slug,
                arguments=arguments,
                user_id=self.user_id,
                connected_account_id=self.connected_account_id,
            )
        except Exception as exc:
            return _safe_json({"status": "error", "action": self.slug, "error": str(exc)})

        normalized = normalize_tool_result(self.slug, _model_dump(raw))
        return _safe_json({"status": "success", "action": self.slug, "result": normalized})


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


def normalize_drive_output(data: Dict[str, Any]) -> Any:
    # Normalize common Drive file listing/metadata to Docs-friendly shape
    def to_item(f: Dict[str, Any]) -> Dict[str, Any]:
        mime = f.get("mimeType") or f.get("mime_type")
        is_doc = (mime == "application/vnd.google-apps.document")
        return {
            "documentId": f.get("id") if is_doc else None,
            "title": f.get("name") or f.get("title"),
            "link": f.get("webViewLink") or f.get("web_view_link") or f.get("alternateLink"),
            "mimeType": mime,
        }

    if isinstance(data.get("files"), list):
        return [to_item(f) for f in data["files"] if isinstance(f, dict)]
    if data.get("id") or data.get("name"):
        return to_item(data)
    return data


def normalize_tool_result(slug: str, data: Dict[str, Any]) -> Any:
    up = (slug or "").upper()
    if up.startswith("GOOGLEDOCS_"):
        return normalize_docs_output(data)
    if up.startswith("GOOGLEDRIVE_"):
        return normalize_drive_output(data)
    return data


def main() -> None:
    args = parse_args()

    composio_api_key = require_env("COMPOSIO_API_KEY")
    openai_api_key = require_env("OPENAI_API_KEY")

    cli_id = args.uuid
    # Resolve CLI ID: accept either a record_id or a user_id
    external_user_id = cli_id
    connected_account_id: Optional[str] = None
    rec = _load_record_by_id(DEFAULT_RECORD_PATH, cli_id)
    if rec:
        # Map to the user_id from the record and try to find a Docs connection for that user
        external_user_id = str(rec.get("user_id") or external_user_id)
        all_recs = _load_all_records(DEFAULT_RECORD_PATH)
        connected_account_id = _find_docs_connected_account_id(all_recs, external_user_id)
    user_prompt = args.prompt or FALLBACK_PROMPT

    composio = Composio(api_key=composio_api_key)
    chat_model = ChatOpenAI(model=args.model, openai_api_key=openai_api_key, temperature=0)
    prompt_template = build_prompt_template(SYSTEM_PROMPT)

    # Docs + Drive toolkit fetch (Drive used for title-based search)
    requested_toolkits = ["GOOGLEDOCS", "GOOGLEDRIVE"]
    raw_tools = composio.tools.get(user_id=external_user_id, toolkits=requested_toolkits)
    if not raw_tools:
        print(f"No Google Docs/Drive tools available for UUID {external_user_id}. Connect Docs and retry.")
        return

    # Prepare tool schemas for LLM + wrappers for execution
    functions: List[Dict[str, Any]] = []
    lc_tools: List[BaseTool] = []
    for t in raw_tools:
        fn = normalize_function_payload(t)
        if not fn:
            continue
        name_up = str(fn.get("name", "")).upper()
        # Allow all GOOGLEDOCS_* and search-safe GOOGLEDRIVE_* (GET/LIST/SEARCH)
        allow = False
        if name_up.startswith("GOOGLEDOCS_"):
            allow = True
        elif name_up.startswith("GOOGLEDRIVE_") and any(k in name_up for k in ("SEARCH", "LIST", "GET")):
            allow = True
        if not allow:
            continue
        functions.append(fn)
        lc_tools.append(
            DocsTool(
                name=str(fn.get("name")),
                description=str(fn.get("description") or ""),
                slug=str(fn.get("name")),
                user_id=external_user_id,
                tools_client=composio.tools,
                connected_account_id=connected_account_id,
            )
        )

    print("[✓] Requested toolkits: " + ", ".join(requested_toolkits))
    print(f"[i] Using user_id: {external_user_id}")
    if connected_account_id:
        print(f"[i] Docs connected_account_id: {connected_account_id}")
    print(f"[i] Exposed Docs tools: {len(lc_tools)}")
    if args.prompt is None:
        print(f"[i] Using fallback prompt: {user_prompt}")

    if not functions or not lc_tools:
        print("No Docs tools resolved after filtering.")
        return

    # Bind function schemas to LLM and run with corresponding wrappers
    agent = create_openai_functions_agent(chat_model, functions, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=lc_tools, verbose=True)

    print(f"Task: {user_prompt}")
    result = agent_executor.invoke({"input": user_prompt})
    output = result.get("output") if isinstance(result, dict) else result
    print(output)


if __name__ == "__main__":
    main()
