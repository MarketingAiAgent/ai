import logging 
import json 

from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.database.supabase import supabase_client, embeddings
from app.core.config import settings 

from app.agents.text_to_sql.__init__ import call_sql_generator
from .state import *
from .helpers import *

logger = logging.getLogger(__name__)

_tavily = TavilySearch(max_results=5)

def run_t2s_agent_with_instruction(state: OrchestratorState, instruction: str): 
    result = call_sql_generator(
        message=instruction, 
        conn_str=state["conn_str"], 
        schema_info=state["schema_info"]
    )
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
        tool = TavilySearch(max_results=max_results, ) if max_results != 5 else _tavily
        out = tool.invoke(query) 
        if isinstance(out, str):
            try:
                parsed_out = json.loads(out)
                out = parsed_out
                
            except json.JSONDecodeError:
                logger.error("Tavily가 일반 에러 문자열을 반환했습니다: %s", out)
                return {"results": [], "error": out}

        results = []
        for r in out.get('results') or []:
            results.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content"),
            })
        return {"results": results}
    except Exception as e:
        logger.error("Tavily 검색 실패: %s", e, "출력:", out)
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

def marketing_trend_search(question: str) -> Dict[str, Any]:
    """
    Supabase 함수 'marketing_vector_search' 호출.
    반환 스키마:
    { "results": [ {"title":..,"chunk_text":..,"text":..,"subtitle":..}, ... ] }
    """
    if not (supabase_client and embeddings):
        return {"results": [], "error": "supabase_client/embeddings not configured"}
    try:
        vec = embeddings.embed_query(question)
        resp = supabase_client.rpc("marketing_vector_search", {"query_vector": vec}).execute()
        out = []
        for it in resp.data or []:
            out.append({
                "title": it.get("title", "제목 없음"),
                "chunk_text": it.get("chunk_text", ""),
                "text": it.get("text", ""),
                "subtitle": it.get("subtitle", ""),
            })
        return {"results": out}
    except Exception as e:
        logger.error("Supabase marketing search 실패: %s", e)
        return {"results": [], "error": str(e)}

def beauty_youtuber_trend_search(question: str) -> Dict[str, Any]:
    """
    Supabase 함수 'beauty_vector_search' 호출.
    스키마는 marketing_trend_search와 동일.
    """
    if not (supabase_client and embeddings):
        return {"results": [], "error": "supabase_client/embeddings not configured"}
    try:
        vec = embeddings.embed_query(question)
        resp = supabase_client.rpc("beauty_vector_search", {"query_vector": vec}).execute()
        out = []
        for it in resp.data or []:
            out.append({
                "title": it.get("title", "제목 없음"),
                "chunk_text": it.get("chunk_text", ""),
                "text": it.get("text", ""),
                "subtitle": it.get("subtitle", ""),
            })
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=settings.GOOGLE_API_KEY)

        SUMMARIZER_PROMPT = """
You are a research assistant. Your task is to analyze the raw text from a tool's output and summarize the key findings that are directly relevant to the user's original question.

User's original question: "{question}"

Raw text from the tool:
---
{raw_text}
---

Based on the raw text, please extract and summarize only the essential information that answers the user's question.
Present the summary in a few concise bullet points in Korean.
If the raw text contains no relevant information to answer the question, just return "관련 정보를 찾을 수 없습니다."
""" 
        prompt = ChatPromptTemplate.from_template(SUMMARIZER_PROMPT)

        summary_chain = prompt | llm
        summary = summary_chain.invoke({
                "question": question,
                "raw_text": out
            }).content
        
        logger.info(f"뷰티 유튜버 확인 완료: {summary}")
            
        return {"results": summary}
    except Exception as e:
        logger.error("Supabase beauty youtuber search 실패: %s", e)
        return {"results": [], "error": str(e)}

STOPWORDS = {
    "트렌드","마케팅","브랜드","제품","상품","리뷰","추천","최신","분석","사례","전략","광고",
    "케이스","뉴스","이슈","영상","채널","컨텐츠","콘텐츠","캠페인","프로모션","업계","시장",
    "올해","이번","최근","핵심","소개","정리","가이드","방법","이유","포인트","사용","활용",
    "and","or","of","for","to","in","on","with","by","the","a","an"
}

HASHTAG_RE = re.compile(r"#([0-9A-Za-z가-힣_]{2,30})")
QUOTED_RE  = re.compile(r"[\"“”‘’']([^\"“”‘’']{2,30})[\"“”‘’']")
HANGUL_TERM_RE = re.compile(r"[가-힣]{2,8}")
EN_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\+]{2,20}")

SEASON_KEYWORDS = {
    "추석":  {"window": "YYYY-09-05~YYYY-09-25", "term": "추석선물"},
    "설날":  {"window": "YYYY-01-15~YYYY-02-20", "term": "설 선물"},
    "여름":  {"window": "YYYY-06-15~YYYY-08-31", "term": "여름휴가"},
    "겨울":  {"window": "YYYY-11-15~YYYY-12-31", "term": "연말/겨울"},
    "봄":    {"window": "YYYY-03-01~YYYY-05-31", "term": "봄 신제품"},
    "가을":  {"window": "YYYY-09-01~YYYY-10-31", "term": "가을 프로모션"},
}

def _normalize_term(t: str) -> str:
    t = t.strip().strip(".,!?()[]{}:;…·・/\\|")
    return t

def _extract_terms(text: str) -> List[str]:
    if not text:
        return []
    terms = []
    # 해시태그 / 따옴표 / 한글 어절 / 영문 토큰
    terms += [m.group(1) for m in HASHTAG_RE.finditer(text)]
    terms += [m.group(1) for m in QUOTED_RE.finditer(text)]
    terms += HANGUL_TERM_RE.findall(text)
    terms += EN_TERM_RE.findall(text)
    out = []
    for t in terms:
        t = _normalize_term(t)
        if not t or t.lower() in STOPWORDS or t in STOPWORDS:
            continue
        # 너무 일반적인 단어 제거
        if len(t) < 2:
            continue
        out.append(t)
    return out

def _rank_terms(corpus: List[str], top_k: int = 8) -> List[str]:
    from collections import Counter
    cnt = Counter([t for t in corpus if t and t.lower() not in STOPWORDS])
    # 한글/영문 섞여 있을 때 길이/빈도 가중 간단 적용
    ranked = sorted(cnt.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    res: List[str] = []
    seen = set()
    for term, _ in ranked:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        res.append(term)
        if len(res) >= top_k:
            break
    return res

def _guess_seasonal_spikes(texts: List[str], year: Optional[int] = None) -> List[Dict[str, str]]:
    """
    문서 텍스트에서 계절/명절 키워드를 발견하면 대략의 윈도우를 반환.
    YYYY는 실제 연도로 교체하지 않고 고정 문자열로 둡니다(소비 측에서 치환 가능).
    """
    found = set()
    spikes: List[Dict[str, str]] = []
    all_txt = " ".join(texts)
    for k, v in SEASON_KEYWORDS.items():
        if k in all_txt:
            if k not in found:
                found.add(k)
                spikes.append({"term": v["term"], "window": v["window"]})
    return spikes

def get_knowledge_snapshot(
    topic: Optional[str] = None,
    *,
    use_web: bool = True,
    use_supabase: bool = True,
    max_results: int = 5,
    scrape_k: int = 3,
) -> Dict[str, Any]:
    """
    외부 신호를 합쳐 '트렌드 스냅샷'을 생성합니다.
    반환 스키마:
    {
      "trending_terms": [str, ...],
      "seasonal_spikes": [{"term": str, "window": "YYYY-MM-DD~YYYY-MM-DD"}, ...],
      "notes": [str, ...],
      "sources": [{"title": str, "url": str}],
      "raw": { "web_search": ..., "scraped_pages": ..., "marketing": ..., "youtuber": ... }  # (옵션)
    }
    """
    # 0) 기본 쿼리
    query = topic or "한국 소비자 트렌드 2025 숏폼 바이럴 마케팅"
    all_texts: List[str] = []
    sources: List[Dict[str, str]] = []
    notes: List[str] = []

    web_search_res = None
    scraped_res = None
    mk_res = None
    yt_res = None

    # 1) 웹 검색
    if use_web:
        web_search_res = run_tavily_search(query, max_results=max_results)
        for r in (web_search_res.get("results") or []):
            title = r.get("title") or ""
            url = r.get("url") or ""
            content = r.get("content") or ""
            sources.append({"title": title, "url": url})
            # 타이틀/스니펫만 먼저 수집
            if title: all_texts.append(title)
            if content: all_texts.append(content)

        # 2) 스크랩(최대 scrape_k개)
        urls = [s["url"] for s in sources if s.get("url")] if sources else []
        if urls:
            scraped_res = scrape_webpages(urls[:scrape_k])
            for d in (scraped_res.get("documents") or []):
                if d.get("content"):
                    all_texts.append(d["content"])

    # 3) Supabase 마케팅/뷰티 인사이트
    if use_supabase:
        mk_res = marketing_trend_search(query)
        yt_res = beauty_youtuber_trend_search(query)
        for item in (mk_res.get("results") or []):
            for k in ("title", "chunk_text", "text", "subtitle"):
                if item.get(k): all_texts.append(item[k])
        for item in (yt_res.get("results") or []):
            for k in ("title", "chunk_text", "text", "subtitle"):
                if item.get(k): all_texts.append(item[k])

    # 4) 용어 추출/랭킹
    corpus_terms: List[str] = []
    for txt in all_texts:
        corpus_terms.extend(_extract_terms(txt))
    trending_terms = _rank_terms(corpus_terms, top_k=8)

    # 5) 시즌 스파이크 감지(간단 키워드 매칭)
    seasonal_spikes = _guess_seasonal_spikes(all_texts)

    # 6) 노트(간단 요약)
    if web_search_res and web_search_res.get("results"):
        notes.append(f"웹 검색 결과 {len(web_search_res['results'])}건 수집")
    if scraped_res and scraped_res.get("documents"):
        notes.append(f"스크랩 문서 {len(scraped_res['documents'])}건 분석")
    if mk_res and mk_res.get("results") is not None:
        notes.append(f"Supabase 마케팅 결과 {len(mk_res['results'])}건")
    if yt_res and yt_res.get("results") is not None:
        notes.append(f"Supabase 뷰티 유튜버 결과 {len(yt_res['results'])}건")

    snapshot = {
        "trending_terms": trending_terms,
        "seasonal_spikes": seasonal_spikes,
        "notes": notes,
        "sources": sources[:max_results],
        # 필요 시 디버깅용 raw를 넘길 수 있으나, 기본은 생략 권장
        # "raw": {
        #   "web_search": web_search_res,
        #   "scraped_pages": scraped_res,
        #   "marketing": mk_res,
        #   "youtuber": yt_res,
        # },
    }

    logger.info("knowledge snapshot: %s", snapshot)
    return snapshot