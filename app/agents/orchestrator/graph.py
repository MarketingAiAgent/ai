from __future__ import annotations

import json
import textwrap
import logging
from typing import List, Optional, Dict, Any, Literal, TypedDict
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from .state import *
from .tools import *
from .helpers import _summarize_history, _today_kr, _normalize_table

logger = logging.getLogger(__name__)

# =========================
# Action-State Adapter (4·5·6)
# =========================

class ActionDecision(TypedDict):
    intent_type: Literal["promotion", "none"]
    status: Literal["start_promotion", "ask_for_slots", "skip"]
    missing_slots: List[str]
    ask_prompts: List[str]
    payload: Dict[str, Any]

def _external_action_state_adapter(history: List[Dict[str, str]],
                                   active_task: Optional[ActiveTask],
                                   user_message: str) -> ActionDecision:
    """
    외부 action_state 모듈 연동 시도.
    없다면 내부 규칙(고정 계약)으로 대체.
    - 휴리스틱 회피: '필수 슬롯'을 명시적 계약으로 강제
    """
    # 1) 외부 모듈이 있으면 사용
    try:
        # 기대 인터페이스: get_action_state(history, active_task, user_message) -> ActionDecision 유사 dict
        from app.agents.actions.action_state import get_action_state  # 사용자가 별도 구현 예정
        dec = get_action_state(history=history, active_task=active_task, user_message=user_message)
        # 최소 필드 보정
        return {
            "intent_type": dec.get("intent_type", "none"),
            "status": dec.get("status", "skip"),
            "missing_slots": dec.get("missing_slots", []),
            "ask_prompts": dec.get("ask_prompts", []),
            "payload": dec.get("payload", {}),
        }
    except Exception as e:
        logger.info("외부 action_state 모듈 미탑재 또는 실패: %s (내부 규칙 사용)", e)

    # 2) 내부 고정 계약(비휴리스틱) - ActiveTask 기반
    #    - active_task가 없으면 프로모션 플로우는 'skip'
    if not active_task or active_task.slots is None:
        return {
            "intent_type": "none",
            "status": "skip",
            "missing_slots": [],
            "ask_prompts": [],
            "payload": {}
        }

    slots: PromotionSlots = active_task.slots

    # 필수 슬롯(명시 계약)
    # target_type별로 필수 슬롯을 정확히 정의 (휴리스틱 아님: 고정 규칙)
    REQUIRED_COMMON = ["objective", "duration", "target_type"]
    REQUIRED_BY_TYPE = {
        "brand_target": ["brand"],
        "category_target": ["target"],
    }

    missing: List[str] = []
    # 공통 필수
    for k in REQUIRED_COMMON:
        if getattr(slots, k) in (None, "", []):
            missing.append(k)
    # 타입별
    ttype = getattr(slots, "target_type", None)
    if ttype in REQUIRED_BY_TYPE:
        for k in REQUIRED_BY_TYPE[ttype]:
            if getattr(slots, k) in (None, "", []):
                missing.append(k)
    else:
        # target_type 자체가 없거나 미지원 값이면 질문 필요
        if "target_type" not in missing:
            missing.append("target_type")

    if missing:
        # 질문 문구는 슬롯명 그대로(표준화). 실제 UX 문구는 별도 레이어에서 바꿀 수 있음.
        ask_prompts = []
        name_map = {
            "objective": "이번 프로모션의 목표(예: 매출 증대, 신규 고객 유입)를 알려주실 수 있을까요?",
            "duration": "프로모션 기간을 알려주실 수 있을까요? (예: 2025-09-01 ~ 2025-09-14)",
            "target_type": "타겟 종류를 선택해 주세요. (brand_target | category_target)",
            "brand": "타겟 브랜드를 알려주실 수 있을까요?",
            "target": "타겟 카테고리/고객군을 알려주실 수 있을까요?",
        }
        for k in missing[:2]:  # 한 번에 1~2개만 묻기
            ask_prompts.append(name_map.get(k, f"{k} 값을 알려주실 수 있을까요?"))
        return {
            "intent_type": "promotion",
            "status": "ask_for_slots",
            "missing_slots": missing,
            "ask_prompts": ask_prompts,
            "payload": {}
        }

    # 모든 필수 슬롯 충족 → 시작 가능
    return {
        "intent_type": "promotion",
        "status": "start_promotion",
        "missing_slots": [],
        "ask_prompts": [],
        "payload": {
            "objective": slots.objective,
            "target_type": slots.target_type,
            "target": slots.target,
            "brand": slots.brand,
            "selected_product": slots.selected_product,
            "duration": slots.duration,
            "product_options": slots.product_options,
        },
    }

def _start_promotion(payload: Dict[str, Any]) -> str:
    """
    실제 생성은 외부 시스템과 연동될 수 있음.
    여기서는 시작 신호만 반환(부작용 없음).
    """
    logger.info("프로모션 생성 시작 (payload): %s", payload)
    # TODO: 외부 오케스트레이터/백오피스 연동 지점
    return "프로모션 생성을 시작하겠습니다. 설정 요약: " + json.dumps(payload, ensure_ascii=False)


# =========================
# Nodes
# =========================

def planner_node(state: OrchestratorState):
    logger.info("--- 1. 🤔 계획 수립 노드 (Planner) 실행 ---")

    parser = PydanticOutputParser(pydantic_object=OrchestratorInstruction)

    history_summary = _summarize_history(state.get("history", []))
    active_task_dump = state['active_task'].model_dump_json() if state.get('active_task') else 'null'
    schema_sig = state.get("schema_info", "") 
    today = _today_kr()

    prompt_template = textwrap.dedent("""
    You are the orchestrator for a marketing agent. Decide what to do this turn using ONLY the provided context.
    You MUST output a JSON that strictly follows: {format_instructions}

    ## Your tools (decide whether to call them this turn)
    - t2s (text-to-SQL): Use when the user's question requires factual data, aggregations, rankings, or trends derived from the company's DB. It returns a TABLE as JSON (rows/columns).
    - knowledge: Use when the user asks for external trend/insight summaries. Do NOT invent DB facts.

    ## Time normalization
    - Convert relative dates to ABSOLUTE ranges with Asia/Seoul timezone. Today is {today}.
      e.g., "올해" => "{year}-01-01 ~ {today}", "지난 달" => first/last day of previous month, etc.
      If ambiguous, choose a reasonable default that does not block execution.

    ## DB schema signature (use as a hint; do not hallucinate columns beyond this):
    {schema_sig}

    ## Conversation summary (last turns):
    {history_summary}

    ## Active task (promotion) snapshot (JSON or null):
    {active_task}

    ## Decision rules (no heuristics, rely on the user's intent and schema above)
    - If the user is asking a question that requires DB facts/aggregations/ranking (e.g., "올해 제일 많이 팔린 상품이 뭐였어?"), you MUST set a clear `t2s_instruction` with explicit metrics, dimensions, sorting, and limit. Prefer revenue/quantity if applicable.
    - If the user is progressing/starting a promotion and required slots are incomplete, do NOT call tools; set `response_generator_instruction` to politely ask 1-2 questions to fill missing slots (target_type, brand/target, duration, objective).
    - If the user requests trend knowledge (not DB facts), use `knowledge_instruction`.
    - If the request is out-of-scope, set both tool instructions to null and provide a helpful guidance in `response_generator_instruction`.
    - Output should be concise and in Korean polite style.

    ## Few-shot
    Q: "올해 제일 많이 팔린 상품이 뭐였어?"
    A: t2s_instruction should describe: "{year}-01-01 ~ {today}" period, aggregate revenue and quantity by product_name, sort by revenue DESC, limit top 3.

    Q: "브랜드 A로 2주 프로모션 만들어줘"
    A: Ask for any missing required slots (target_type, duration, etc.) in Korean polite style.

    Q: "요즘 숏폼 트렌드 뭐야?"
    A: Use knowledge_instruction to fetch recent short-form/UGC trend highlights.

    User Message: "{user_message}"
    """)

    year = today[:4]
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
        api_key=settings.GOOGLE_API_KEY
    )

    instructions = (prompt | llm | parser).invoke({
        "user_message": state['user_message'],
        "history_summary": history_summary,
        "active_task": active_task_dump,
        "schema_sig": schema_sig,
        "today": today,
        "year": year
    })

    return {"instructions": instructions}

def action_state_node(state: OrchestratorState):
    """
    --- 2. 📋 액션 상태 확인 노드 (Action-State) ---
    4) 프로모션 관련 여부 판단
    5) 필수 슬롯 충족 시 'start_promotion'
    6) 불충분 시 'ask_for_slots' 질문 생성

    - state 스키마 변경 없음
    - 외부 모듈 존재 시 우선 사용, 없으면 고정 계약으로 동작
    """
    logger.info("--- 2. 📋 액션 상태 확인 노드 (Action-State) 실행 ---")
    decision = _external_action_state_adapter(
        history=state.get("history", []),
        active_task=state.get("active_task"),
        user_message=state.get("user_message", "")
    )
    logger.info("Action decision: %s", decision)
    return {"tool_results": {"action": decision}}

def tool_executor_node(state: OrchestratorState):
    logger.info("--- 3. 🔨 툴 실행 노드 (Tool Executor) 실행 ---")

    instructions = state.get("instructions")
    existing_results = dict(state.get("tool_results") or {})
    if not instructions or (not instructions.t2s_instruction and not instructions.knowledge_instruction):
        logger.info("호출할 툴이 없습니다.")
        return {"tool_results": existing_results or None}

    tool_results: Dict[str, Any] = {}

    with ThreadPoolExecutor() as executor:
        futures = {}
        if instructions.t2s_instruction:
            futures[executor.submit(run_t2s_agent, state)] = "t2s"
        if instructions.knowledge_instruction:
            futures[executor.submit(run_knowledge_agent, instructions.knowledge_instruction)] = "knowledge"

        for future in futures:
            tool_name = futures[future]
            try:
                tool_results[tool_name] = future.result()
            except Exception as e:
                logger.error("%s 툴 실행 중 에러 발생: %s", tool_name, e)
                tool_results[tool_name] = {"error": str(e)}

    merged = {**existing_results, **tool_results} if existing_results else tool_results
    return {"tool_results": merged}

def response_generator_node(state: OrchestratorState):
    logger.info("--- 4. 🗣️ 응답 생성 노드 (Response Generator) 실행 ---")

    instructions = state.get("instructions")
    tool_results = state.get("tool_results") or {}

    instructions_text = (
        instructions.response_generator_instruction
        if instructions and instructions.response_generator_instruction
        else "사용자 요청에 맞춰 정중하고 간결하게 답변해 주세요."
    )

    t2s_payload = tool_results.get("t2s")
    t2s_table = _normalize_table(t2s_payload) if t2s_payload else None

    action_decision = tool_results.get("action")  # action_state_node 결과 그대로
    knowledge_snippet = tool_results.get("knowledge") if isinstance(tool_results.get("knowledge"), str) else None

    prompt_tmpl = textwrap.dedent("""
    당신은 마케팅 오케스트레이터의 최종 응답 생성기입니다.
    아래 입력만을 근거로 **한국어 존댓말**로 한 번에 완성된 답변을 작성해 주세요.
    내부 도구명이나 시스템 세부 구현은 언급하지 않습니다.

    [입력 설명]
    - instructions_text: 이번 턴에서 어떤 톤/방향으로 응답해야 하는지에 대한 상위 지시.
    - action_decision: 프로모션 관련 의사결정 결과 오브젝트(존재할 수도, 없을 수도 있음).
        예) {{
          "intent_type": "promotion" | "none",
          "status": "start_promotion" | "ask_for_slots" | "skip",
          "missing_slots": [...],
          "ask_prompts": [...],
          "payload": {{...}}
        }}
    - t2s_table: 회사 DB로부터 생성된 질의 결과 표. 형식은 {{
        "rows": [{{...}}, ...],
        "columns": ["colA", "colB", ...],
        "row_count": N
      }} 또는 None.
      ※ 표가 있다면 표 **내용만** 사용하여 사실을 말해 주세요. 새로운 수치/사실 창작 금지.
      ※ 표가 매우 크더라도 **미리보기 용도로 상위 10행만** 마크다운 테이블로 보여 주세요(열 순서는 columns 기준).
    - knowledge_snippet: 외부 트렌드 요약 문자열(있을 수도, 없을 수도 있음).

    [작성 지침]
    1) instructions_text를 최상위 가이드로 삼아 톤과 범위를 결정합니다.
    2) action_decision이 존재하면:
       - status=="ask_for_slots"인 경우, ask_prompts에 들어있는 질문을 **정중하게 1~2문장으로** 자연스럽게 물어보세요.
       - status=="start_promotion"인 경우, payload의 핵심 설정을 **짧게 요약**하고 시작을 확인해 주세요.
       - "skip"이거나 없으면 무시합니다.
    3) t2s_table이 존재하면:
       - 표를 근거로 핵심 포인트를 **간결히 요약**하세요(예: 눈에 띄는 항목/증감/비중 등).
       - 표 **미리보기(최대 10행)**를 마크다운 표로 포함하세요.
       - 열 이름이 표준화되어 있지 않아도 추측하지 말고 **있는 값만** 사용하세요.
    4) knowledge_snippet이 있으면 "참고 트렌드"로 1~2줄만 덧붙여 주세요.
    5) 데이터가 부족하거나 결론이 어려우면, 정중하게 한 문장으로 한계를 밝히고 다음 액션을 제안하세요.
    6) 전반적으로 **간결하고 구조화**(불릿/짧은 단락)하며, 한국어 존댓말을 유지하세요.

    [입력 데이터]
    - instructions_text:
    {instructions_text}

    - action_decision (JSON):
    {action_decision_json}

    - t2s_table (JSON):
    {t2s_table_json}

    - knowledge_snippet:
    {knowledge_snippet}
    """)

    action_json = json.dumps(action_decision, ensure_ascii=False) if action_decision is not None else "null"
    table_json = json.dumps(t2s_table, ensure_ascii=False) if t2s_table is not None else "null"

    prompt = ChatPromptTemplate.from_template(prompt_tmpl)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=settings.GOOGLE_API_KEY
    )

    final_text = (prompt | llm).invoke({
        "instructions_text": instructions_text,
        "action_decision_json": action_json,
        "t2s_table_json": table_json,
        "knowledge_snippet": knowledge_snippet or ""
    })

    final_response = getattr(final_text, "content", None) or str(final_text)

    logger.info(f"최종 결과(L):\n{final_response}")

    history = state.get("history", [])
    history.append({"role": "user", "content": state.get("user_message", "")})
    history.append({"role": "assistant", "content": final_response})

    return {"history": history, "user_message": "", "output": final_response}


# =========================
# Graph
# =========================
workflow = StateGraph(OrchestratorState)

workflow.add_node("planner", planner_node)
workflow.add_node("action_state", action_state_node)       
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("response_generator", response_generator_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "action_state")               
workflow.add_edge("action_state", "tool_executor")         
workflow.add_edge("tool_executor", "response_generator")
workflow.add_edge("response_generator", END)

orchestrator_app = workflow.compile()
