import logging 
import json 

from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WebBaseLoader

# from app.integrations.supabase import supabase_client, embeddings

from app.agents.text_to_sql.__init__ import call_sql_generator
from .state import *
from .helpers import *

logger = logging.getLogger(__name__)

_tavily = TavilySearch(max_results=5)

def run_t2s_agent_with_instruction(state: OrchestratorState):
    result = call_sql_generator(message=state.instruction, conn_str=state.conn_str, schema_info=state.schema_sig)
    table = result.get("data_json")
    if isinstance(table, str):
        try:
            table = json.loads(table)
        except Exception:
            table = {"rows": [], "columns": [], "row_count": 0}
    return ensure_table_payload(table)

def run_tavily_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    간단한 웹 검색. 결과 스키마는 아래 형태로 고정:
    { "results": [ {"title":..., "url":..., "content":...}, ... ] }
    """
    try:
        tool = TavilySearch(max_results=max_results) if max_results != 5 else _tavily
        out = tool.invoke({"query": query})  # langchain tool 규격
        # out 예시는 [{"url":"..","content":"..","title":".."}, ...]
        results = []
        for r in out or []:
            results.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content"),
            })
        return {"results": results}
    except Exception as e:
        logger.error("Tavily 검색 실패: %s", e)
        return {"results": [], "error": str(e)}

def scrape_webpages(urls: List[str]) -> Dict[str, Any]:
    """
    주어진 URL 목록에서 본문을 가져와 합칩니다.
    반환 스키마:
    { "documents": [ {"source": <title or url>, "content": <text>}, ... ] }
    """
    if not urls:
        return {"documents": []}
    if WebBaseLoader is None:
        return {"documents": [], "error": "WebBaseLoader not available"}

    try:
        loader = WebBaseLoader(
            web_path=urls,
            header_template={
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/102.0.0.0 Safari/537.36"),
            },
        )
        docs = loader.load()
        items = []
        for d in docs:
            src = d.metadata.get("title") or d.metadata.get("source") or d.metadata.get("url") or ""
            items.append({"source": src, "content": d.page_content})
        return {"documents": items}
    except Exception as e:
        logger.error("스크래핑 실패: %s", e)
        return {"documents": [], "error": str(e)}