# agent/nodes/plan_option_prompts_node.py
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.agents_v2.orchestrator.state import AgentState, PromotionSlots, OptionToolPlans, ToolChoice, OptionPlanningNote, SQLPlan, OptionWebPlan, AllowedSources

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------

def _build_messages(history: List[str], slots: PromotionSlots) -> List[tuple]:
    """
    이 노드는 '사고 → 실행 인스트럭션'을 만든다.
    - 컬럼명/테이블명 강제 금지 (T2S가 스키마로 스스로 선택)
    - 단, downstream 연결을 위해 출력 표의 첫 컬럼 이름만 `name`으로 요구 (표시용 타겟명)
    - 사람이 읽을 설명용 텍스트 컬럼 `rationale`은 가능하면 포함(없으면 비워도 허용)
    """
    history = (history or [])[-4:]

    system = (
        "당신은 마케팅 옵션 소싱을 위한 **계획 플래너**입니다. "
        "입력(대화 히스토리, PromotionSlots)을 보고 다음을 수행하세요:\n"
        "1) 구조화 사고(PlanningNote)를 작성한다. (자유서술 금지, 반드시 필드에만 기록)\n"
        "2) 위 계획을 바탕으로 Text-to-SQL 에이전트가 그대로 실행할 수 있는 자연어 인스트럭션(SQLPlan.instruction)을 만든다.\n"
        "   - 데이터베이스의 실제 스키마/컬럼명은 모른다. 컬럼명 지시 금지.\n"
        "   - 대신 '표의 의미'와 '필터/집계/정렬의 개념'을 분명히 설명한다.\n"
        "   - 출력 표는 다음 형식을 권장: 첫 컬럼 이름은 반드시 name(타겟 표시용). 그 외에는 스키마에서 가장 적합한 지표/수치를 1~3개 선택.\n"
        "   - 각 행에 짧은 사람이 읽을 설명 텍스트 rationale 컬럼을 가능하면 포함(스키마에 없으면 비워도 됨).\n"
        "   - 상위 {{top_k}}개만 포함하도록 제한하는 문구를 포함.\n"
        "3) WebPlan.query는 간결한 1줄 검색어로 작성. scope/period/audience를 녹여 검색성 높이기.\n"
        "4) JSON 외 텍스트 절대 금지.\n"
    )

    # Few-shot: 브랜드/제품 각 1건 (컬럼명 지시 없이 개념 중심으로 작성)
    fewshot = [
        # --- 브랜드 예시 ---
        ("human", json.dumps({
            "history": ["20대 대상 프로모션 해볼래!"],
            "slots": {{"scope": "브랜드", "period": "다음 달 4주", "audience": "20대", "target": None, "KPI": None, "concept": None}}
        }, ensure_ascii=False)),
        ("ai", json.dumps({
            "tool_choice": "both",
            "planning": {
                "goal": "20대 대상, 브랜드 기준 타겟 후보 상위 3개 도출",
                "needed_table": "행=브랜드; 열=브랜드명과 프로모션 의사결정에 유용한 핵심 지표(최근 성과/성장/충성/신규 등) 요약",
                "filters": ["기간: 다음 달 4주", "오디언스: 20대 관련 지표가 있으면 반영"],
                "metrics_preference": ["최근 매출/주문/방문 지표", "직전기간 대비 변화율", "반복구매 또는 신규유입 관련 수치"],
                "ranking_logic": "프로모션 적합도가 높다고 판단되는 지표의 결합 점수로 상위 3개 선정(스키마에 맞게 자율 선정)",
                "notes": "스키마 명시 금지. 출력 표 첫 컬럼은 name(브랜드 표시), 설명 텍스트는 rationale 컬럼에 요약."
            },
            "sql": {
                "enabled": True,
                "instruction": (
                    "다음 달 4주를 대상으로, 브랜드 단위로 표를 만들어 주세요. "
                    "표의 첫 번째 열 이름은 반드시 name이며, 각 행에 브랜드 표시 이름을 넣어 주세요. "
                    "그 외 열은 데이터베이스 스키마에서 프로모션 의사결정에 가장 유용한 지표(예: 최근 성과, 성장, 충성/신규 관련)를 1~3개 선택하여 포함해 주세요. "
                    "가능하다면 각 행에 해당 브랜드가 20대 대상 프로모션에 적합한 이유를 간단히 서술하는 rationale 텍스트 열을 추가해 주세요(없으면 비워도 됩니다). "
                    "지표가 여러 개일 경우, 스키마에 맞는 방식으로 적절히 결합/정렬하여 적합도가 높은 상위 3개 브랜드만 포함해 주세요."
                ),
                "queries": [],
                "top_k": 3
            },
            "web": {
                "enabled": True,
                "query": "20대 브랜드 트렌드 다음 달 캠페인 아이디어 숏폼",
                "queries": [],
                "use_sources": ["supabase_marketing","supabase_beauty","tavily"],
                "top_k": 3,
                "scrape_k": 2
            }
        }, ensure_ascii=False)),

        # --- 제품 예시 ---
        ("human", json.dumps({
            "history": ["브랜드보단 제품 기준으로 가자."],
            "slots": {{"scope": "제품", "period": "이번 달 2주", "audience": None, "target": None, "KPI": None, "concept": None}}
        }, ensure_ascii=False)),
        ("ai", json.dumps({
            "tool_choice": "both",
            "planning": {
                "goal": "제품 기준 타겟 후보 상위 3개 도출",
                "needed_table": "행=제품; 열=제품명과 최근 성과/성장/수요 징후 등 의사결정 핵심 지표 요약",
                "filters": ["기간: 이번 달 2주"],
                "metrics_preference": ["최근 판매 또는 조회/장바구니 등 수요 지표", "직전기간 대비 변화율", "반복구매 또는 신규유입 관련 수치"],
                "ranking_logic": "스키마에서 사용 가능한 지표로 종합 적합도 산정 후 상위 3개",
                "notes": "출력 표 첫 컬럼은 name(제품 표시), 가능하면 rationale 컬럼 포함."
            },
            "sql": {
                "enabled": True,
                "instruction": (
                    "이번 달 2주를 대상으로, 제품 단위 표를 만들어 주세요. "
                    "표의 첫 번째 열 이름은 반드시 name이며, 각 행에 제품 표시 이름을 넣어 주세요. "
                    "그 외 열은 스키마에서 사용 가능한 핵심 지표(최근 판매/수요, 변화율, 반복/신규 관련) 중 1~3개를 선택하여 포함해 주세요. "
                    "가능하면 각 행에 해당 제품이 프로모션 타겟으로 적합한 이유를 요약한 rationale 텍스트 열을 추가해 주세요(없으면 비워도 됩니다). "
                    "스키마에 맞는 방식으로 적합도를 산정해 상위 3개 제품만 포함해 주세요."
                ),
                "queries": [],
                "top_k": 3
            },
            "web": {
                "enabled": True,
                "query": "제품 트렌드 인기 제품 이번 달 2주 리뷰 숏폼",
                "queries": [],
                "use_sources": ["supabase_marketing","supabase_beauty","tavily"],
                "top_k": 3,
                "scrape_k": 0
            }
        }, ensure_ascii=False)),
    ]

    user = ("human", json.dumps({
        "history": history,
        "slots": slots.model_dump(),
    }, ensure_ascii=False))

    return [("system", system), *fewshot, user]


# ---------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------

def plan_option_prompts_node(state: AgentState) -> AgentState:
    history = state.history or []
    slots = state.promotion_slots

    if not slots:
        slots = PromotionSlots()

    messages = _build_messages(history, slots)
    parser = PydanticOutputParser(pydantic_object=OptionToolPlans)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=settings.GOOGLE_API_KEY,
    )

    try:
        plans: OptionToolPlans = (llm | parser).invoke(messages)
    except Exception:
        logger.exception("[plan_option_prompts_node] LLM 실패 → 최소 폴백 계획 생성")
        scope_kw = "브랜드" if slots.scope == "브랜드" else "제품"
        plans = OptionToolPlans(
            tool_choice="both",
            planning=OptionPlanningNote(
                goal=f"{scope_kw} 기준 상위 3개 후보 도출",
                needed_table=f"행={scope_kw}; 열={scope_kw} 표시명(name)과 스키마에서 가장 유용한 핵심 지표 1~3개",
                filters=[f"기간: {slots.period or '최근 기간'}"] + ([f"오디언스: {slots.audience}"] if slots.audience else []),
                metrics_preference=["최근 성과", "직전기간 대비 변화", "반복/신규 관련"],
                ranking_logic="사용 가능한 지표로 적합도 산정 후 상위 3개",
                notes="첫 열 이름은 name, 가능하면 rationale 열 포함",
            ),
            sql=SQLPlan(
                enabled=True,
                instruction=(
                    f"{slots.period or '최근 기간'}을 대상으로, {scope_kw} 단위 표를 만들어 주세요. "
                    "표의 첫 번째 열 이름은 반드시 name이며, 각 행에 표시 이름을 넣어 주세요. "
                    "그 외 열은 스키마에서 선택 가능한 핵심 지표 1~3개를 포함해 주세요. "
                    "가능하면 각 행에 간단 설명 텍스트 rationale 열을 추가해 주세요(없으면 비워도 됩니다). "
                    "적합도가 높은 상위 3개만 포함해 주세요."
                ),
                queries=[],
                top_k=3
            ),
            web=OptionWebPlan(
                enabled=True,
                query=f"{slots.audience or ''} {scope_kw} 트렌드 {slots.period or ''}".strip(),
                queries=[],
                use_sources=["supabase_marketing","supabase_beauty","tavily"],
                top_k=3,
                scrape_k=2 if slots.scope == "브랜드" else 0
            )
        )

    plans.sql.top_k = max(1, min(5, plans.sql.top_k or 3))
    plans.web.top_k = max(1, min(5, plans.web.top_k or 3))
    plans.web.scrape_k = max(0, min(5, plans.web.scrape_k or 0))

    if not plans.sql.instruction and plans.sql.queries:
        plans.sql.instruction = plans.sql.queries[0]
    if not plans.web.query and plans.web.queries:
        plans.web.query = plans.web.queries[0]

    return state.model_copy(update={"tool_plans": plans})
