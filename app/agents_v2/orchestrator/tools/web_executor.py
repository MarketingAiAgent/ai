# orchestrator/tools/web_executor.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.database.supabase import supabase_client, embeddings
from app.core.config import settings

logger = logging.getLogger(__name__)

_TAVILY_DEFAULT = TavilySearch(max_results=5)
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/102.0.0.0 Safari/537.36"
)


# ----- thin utilities kept public (호환성 유지) -----

def run_tavily_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    out: Any = None
    try:
        tool = _TAVILY_DEFAULT if max_results == 5 else TavilySearch(max_results=max_results)
        out = tool.invoke(query)
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except json.JSONDecodeError:
                return {"results": [], "error": str(out)}
        results = []
        for r in (out.get("results") or []):
            results.append({"title": r.get("title"), "url": r.get("url"), "content": r.get("content")})
        return {"results": results}
    except Exception as e:
        logger.exception("Tavily search failed: %s | raw=%r", e, out)
        return {"results": [], "error": str(e)}


def scrape_webpages(urls: List[str]) -> Dict[str, Any]:
    if not urls:
        return {"documents": []}
    try:
        loader = WebBaseLoader(web_path=urls, header_template={"User-Agent": _USER_AGENT})
        docs = loader.load()
        items = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            src = meta.get("title") or meta.get("source") or meta.get("url") or ""
            items.append({"source": src, "content": getattr(d, "page_content", "")})
        return {"documents": items}
    except Exception as e:
        logger.exception("Scraping failed: %s", e)
        return {"documents": [], "error": str(e)}


def _vector_search(function_name: str, question: str) -> Dict[str, Any]:
    if not (supabase_client and embeddings):
        return {"results": [], "error": "supabase_client/embeddings not configured"}
    try:
        vec = embeddings.embed_query(question)
        resp = supabase_client.rpc(function_name, {"query_vector": vec}).execute()
        out = []
        for it in (resp.data or []):
            out.append({
                "title": it.get("title", "제목 없음"),
                "chunk_text": it.get("chunk_text", ""),
                "text": it.get("text", ""),
                "subtitle": it.get("subtitle", ""),
            })
        return {"results": out}
    except Exception as e:
        logger.exception("Supabase vector search '%s' failed: %s", function_name, e)
        return {"results": [], "error": str(e)}


def marketing_trend_search(question: str) -> Dict[str, Any]:
    return _vector_search("marketing_vector_search", question)


def beauty_youtuber_trend_search(question: str, summarize: bool = True) -> Dict[str, Any]:
    base = _vector_search("beauty_vector_search", question)
    if not summarize or base.get("error"):
        return base
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=settings.GOOGLE_API_KEY)
        prompt = ChatPromptTemplate.from_template(
            "사용자 질문: {q}\n\n다음 원문을 바탕으로 핵심만 한국어 불릿으로 간단 요약하세요.\n---\n{raw}\n---"
        )
        summary = (prompt | llm).invoke({"q": question, "raw": base.get("results", [])}).content
        return {"results": summary}
    except Exception as e:
        logger.exception("Beauty youtuber summarization failed: %s", e)
        return base


# ----- executors (플래너 지시사항만 실행) -----

def execute_web_from_plan(web_plan) -> List[Dict[str, str]]:
    """
    web_plan: WebPlan BaseModel
    반환: [{"name": str, "signal": str, "source": str}]
    """
    if not web_plan or not web_plan.enabled:
        return []

    q = web_plan.query or (web_plan.queries[0] if web_plan.queries else None)
    if not q:
        return []
    
    use_sources = web_plan.use_sources or ["supabase_marketing", "supabase_beauty", "tavily"]
    top_k = web_plan.top_k or 3
    scrape_k = web_plan.scrape_k or 0

    docs: List[Dict[str, str]] = []

    if "supabase_marketing" in use_sources:
        mk = marketing_trend_search(q)
        for r in (mk.get("results", [])):
            docs.append({"title": r.get("title", ""), "url": "", "content": (r.get("chunk_text") or r.get("text") or "")})

    if "supabase_beauty" in use_sources:
        yt = beauty_youtuber_trend_search(q, summarize=False)
        for r in (yt.get("results", [])):
            docs.append({"title": r.get("title", ""), "url": "", "content": (r.get("chunk_text") or r.get("text") or "")})

    if "tavily" in use_sources:
        web = run_tavily_search(q, max_results=max(5, top_k * 2))
        for r in (web.get("results", [])):
            docs.append({"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")})
        if scrape_k > 0:
            urls = [d["url"] for d in docs if d.get("url")]
            if urls:
                scraped = scrape_webpages(urls[:scrape_k])
                for d in (scraped.get("documents", [])):
                    docs.append({"title": d.get("source", ""), "url": "", "content": d.get("content", "")})

    # 최소 정규화만 수행
    out: List[Dict[str, str]] = []
    for d in docs:
        title = (d.get("title", "")).strip()
        content = (d.get("content", "")).strip()
        source = (d.get("url", "")).strip()
        name = title or (content[:40] + "…") if content else "untitled"
        signal = content[:180] + ("…" if len(content) > 180 else "")
        out.append({"name": name, "signal": signal, "source": source})

    return out[: max(1, top_k * 2)]
