from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents_v2.orchestrator.state import AgentState
from app.core.config import settings

logger = logging.getLogger(__name__)

Choice = Literal["t2s", "web", "both", "none"]
AllowedSources = Literal["supabase_marketing", "supabase_beauty", "tavily"]


# -------------------- Schemas --------------------

class PlanningNote(BaseModel):
    """스키마/컬럼명을 모른다는 전제의 CoT 요약(구조화)."""
    user_intent: str
    needed_table: str                  # 예: "행=요일, 지표=전환 관련 핵심 요약"
    filters: List[str] = Field(default_factory=list)        # 기간/세그먼트 등(스키마 명시 금지)
    granularity: Optional[str] = None  # 일/주/월, 캠페인/채널 등
    comparison: Optional[str] = None   # 전년동기/직전기간 등
    why_t2s: Optional[str] = None
    why_web: Optional[str] = None

class T2SPlan(BaseModel):
    enabled: bool = True
    instruction: Optional[str] = None  
    top_rows: int = 100
    visualize: bool = True
    viz_hint: Optional[str] = None     

class WebPlan(BaseModel):
    enabled: bool = True
    query: Optional[str] = None
    queries: List[str] = Field(default_factory=list)
    use_sources: List[AllowedSources] = Field(
        default_factory=lambda: ["supabase_marketing", "supabase_beauty", "tavily"]
    )
    top_k: int = 5
    scrape_k: int = 0

class QAPlan(BaseModel):
    choice: Choice
    planning: PlanningNote
    t2s: T2SPlan
    web: WebPlan


# -------------------- Prompt --------------------

def _build_messages(history: List[str], question: str) -> List[tuple]:
    system = (
        "당신은 마케팅 데이터/트렌드 Q&A를 위한 **플래너**입니다.\n"
        "입력(히스토리+질문)을 보고 아래 JSON 스키마로만 답하세요.\n"
        "{\n"
        '  "choice": "t2s" | "web" | "both" | "none",\n'
        '  "planning": {\n'
        '    "user_intent": "핵심 의도",\n'
        '    "needed_table": "원하는 표의 의미(행/지표)와 관점",\n'
        '    "filters": ["기간/세그먼트 등(스키마명 금지)"],\n'
        '    "granularity": "일/주/월 등(해당 시)",\n'
        '    "comparison": "전년동기/직전기간 등(해당 시)",\n'
        '    "why_t2s": "T2S 이유(선택)",\n'
        '    "why_web": "외부 지식 이유(선택)"\n'
        "  },\n"
        '  "t2s": {\n'
        '    "enabled": true/false,\n'
        '    "instruction": "스키마 모름. 표의 의미/필터/정렬/상위 제한을 자연어로 지시. 첫 컬럼은 name 권장.",\n'
        '    "top_rows": 100,\n'
        '    "visualize": true/false,\n'
        '    "viz_hint": "선/막대/파이 등 힌트(선택)"\n'
        "  },\n"
        '  "web": {\n'
        '    "enabled": true/false,\n'
        '    "query": "간결 검색어",\n'
        '    "queries": ["대체 검색어(선택)"],\n'
        '    "use_sources": ["supabase_marketing","supabase_beauty","tavily"],\n'
        '    "top_k": 5,\n'
        '    "scrape_k": 0\n'
        "  }\n"
        "}\n"
        "규칙: 컬럼/테이블 이름을 추정/강제하지 말고, 표의 의미·필터·정렬의 개념만 지시하세요. JSON 외 텍스트 금지."
    )

    fewshot = [
        ("human", json.dumps({
            "history": [],
            "question": "요일별 전환율 비교해줘. 차트도 부탁."
        }, ensure_ascii=False)),
        ("ai", json.dumps({
            "choice": "t2s",
            "planning": {
                "user_intent": "요일별 전환율 비교 및 시각화",
                "needed_table": "행=요일, 지표=전환율과 관련 핵심 지표",
                "filters": ["기간: 지난달"],
                "granularity": "요일",
                "comparison": None,
                "why_t2s": "내부 전환 데이터 필요"
            },
            "t2s": {
                "enabled": True,
                "instruction": "지난달 데이터를 대상으로 요일 단위 표를 만들어 주세요. "
                               "표의 첫 컬럼은 name(요일)로 하고, 전환율과 핵심 지표 1~3개를 포함해 주세요. "
                               "전환율 기준으로 내림차순 정렬하고 상위 7개(모든 요일)만 포함해 주세요.",
                "top_rows": 100,
                "visualize": True,
                "viz_hint": "요일별 막대"
            },
            "web": {"enabled": False, "query": None, "queries": [], "use_sources": ["supabase_marketing","supabase_beauty","tavily"], "top_k": 5, "scrape_k": 0}
        }, ensure_ascii=False)),

        # Web만
        ("human", json.dumps({
            "history": ["여름 성수기 대비 캠페인 준비 중"],
            "question": "뷰티 인플루언서 트렌드 핵심 키워드 알려줘"
        }, ensure_ascii=False)),
        ("ai", json.dumps({
            "choice": "web",
            "planning": {
                "user_intent": "뷰티 인플루언서 트렌드 키워드 파악",
                "needed_table": "표 필요 없음",
                "filters": ["기간: 최근"],
                "granularity": None,
                "comparison": None,
                "why_web": "외부 트렌드/벡터 인덱스 필요"
            },
            "t2s": {"enabled": False, "instruction": None, "top_rows": 100, "visualize": False, "viz_hint": None},
            "web": {
                "enabled": True,
                "query": "뷰티 인플루언서 최신 트렌드 키워드",
                "queries": [],
                "use_sources": ["supabase_marketing","supabase_beauty","tavily"],
                "top_k": 5,
                "scrape_k": 2
            }
        }, ensure_ascii=False)),

        # Both
        ("human", json.dumps({
            "history": ["신규 고객 유입이 줄었어."],
            "question": "채널별 신규구매수 추이랑, 업계에서 요즘 뭐가 먹히는지 같이 알려줘"
        }, ensure_ascii=False)),
        ("ai", json.dumps({
            "choice": "both",
            "planning": {
                "user_intent": "내부 신규구매수 추이 분석 + 외부 트렌드 보강",
                "needed_table": "행=채널/기간, 지표=신규구매수와 관련 핵심 지표",
                "filters": ["기간: 최근 8주"],
                "granularity": "주",
                "comparison": "직전 기간 대비",
                "why_t2s": "내부 채널 성과 확인",
                "why_web": "업계 최신 전술 참고"
            },
            "t2s": {
                "enabled": True,
                "instruction": "최근 8주를 대상으로 채널-주 단위 표를 만들어 주세요. "
                               "첫 컬럼은 name(채널)로 하고, 신규구매수 및 관련 핵심 지표 1~3개 포함. "
                               "직전 주 대비 변화를 포함해 추이 해석이 가능하도록 정렬/제한을 적용해 주세요(상위 채널 위주).",
                "top_rows": 200,
                "visualize": True,
                "viz_hint": "채널별 주간 추이 선형"
            },
            "web": {
                "enabled": True,
                "query": "이커머스 신규고객 유입 전술 2025 트렌드 사례",
                "queries": [],
                "use_sources": ["supabase_marketing","supabase_beauty","tavily"],
                "top_k": 5,
                "scrape_k": 2
            }
        }, ensure_ascii=False)),
    ]

    user = ("human", json.dumps({"history": history[-4:], "question": question}, ensure_ascii=False))
    return [("system", system), *fewshot, user]


# -------------------- Node --------------------

def qa_plan_node(state: AgentState) -> AgentState:
    logger.info("===== 📝 QA 플래너 노드 실행 =====")

    history = state.history or []
    question = state.user_message or ""

    prompt = ChatPromptTemplate.from_messages(_build_messages(history, question))
    parser = PydanticOutputParser(pydantic_object=QAPlan)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=settings.GOOGLE_API_KEY)

    try:
        plan: QAPlan = (prompt | llm | parser).invoke({})
        plan.t2s.top_rows = max(10, min(1000, plan.t2s.top_rows or 100))
        plan.web.top_k = max(1, min(10, plan.web.top_k or 5))
        plan.web.scrape_k = max(0, min(5, plan.web.scrape_k or 0))
        if not plan.web.query and plan.web.queries:
            plan.web.query = plan.web.queries[0]

        logger.info(f"결과: {plan.model_dump()}")
        logger.info(f"===== 📝 QA 플래너 노드 실행 완료 =====")

        return {"qa_plan": plan.model_dump()}
    except Exception:
        logger.exception("[qa_plan_node] LLM 실패 → 안전 폴백")
        fallback = QAPlan(
            choice="t2s",
            planning=PlanningNote(
                user_intent="기본 지표 조회",
                needed_table="행=핵심 관점, 지표=핵심 수치",
                filters=["기간: 최근"],
            ),
            t2s=T2SPlan(
                enabled=True,
                instruction="가장 최근 기간의 핵심 지표 표를 만들어 주세요. 첫 컬럼은 name으로 표시명을 넣고 상위 20행만 포함해 주세요.",
                top_rows=100,
                visualize=True,
                viz_hint=None
            ),
            web=WebPlan(enabled=False),
        )
        return {"qa_plan": fallback.model_dump()}
