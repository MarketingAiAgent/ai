from __future__ import annotations

import json
import textwrap
import logging
from typing import List, Optional, Dict, Any, Literal, TypedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from app.database.promotion_slots import update_state
from app.agents.promotion.state import get_action_state
from .state import *
from .tools import *
from .helpers import *

logger = logging.getLogger(__name__)

# ===== Nodes =====

def _merge_slots(state: OrchestratorState, updates: Dict[str, Any]) -> PromotionSlots:
    current = (state.get("active_task").slots if state.get("active_task") and state.get("active_task").slots else PromotionSlots())
    base = current.model_dump()
    for k, v in updates.items():
        if v not in (None, "", []):
            base[k] = v
    merged = PromotionSlots(**base)
    if state.get("active_task"):
        state["active_task"].slots = merged
    return merged

def slot_extractor_node(state: OrchestratorState):
    logger.info("--- 2. 슬롯 추출/저장 노드 실행 ---")
    user_message = state.get("user_message", "")
    thread_id = state["thread_id"]

    parser = PydanticOutputParser(pydantic_object=PromotionSlotUpdate)
    prompt_tmpl = textwrap.dedent("""
    아래 한국어 사용자 메시지에서 **프로모션 슬롯 값**을 추출해 주세요.
    - 존재하는 값만 채우고, 없으면 null로 두세요.
    - target_type은 "brand_target" 또는 "category_target" 중 하나로만.
    - 날짜/기간은 원문 그대로 문자열로 유지.
    - 출력은 반드시 다음 JSON 스키마를 따르세요:
      {format_instructions}

    [사용자 메시지]
    {user_message}
    """)
    prompt = ChatPromptTemplate.from_template(
        prompt_tmpl,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
        api_key=settings.GOOGLE_API_KEY
    )
    parsed: PromotionSlotUpdate = (prompt | llm | parser).invoke({"user_message": user_message})

    updates = {k: v for k, v in parsed.model_dump().items() if v not in (None, "", [])}
    if not updates:
        logger.info("슬롯 업데이트 없음")
        return {}

    try:
        update_state(thread_id, updates)
        logger.info("Mongo 상태 업데이트: %s", updates)
    except Exception as e:
        logger.error("Mongo 업데이트 실패: %s", e)

    merged = _merge_slots(state, updates)
    logger.info("State 슬롯 병합: %s", merged.model_dump())
    return {"tool_results": {"slot_updates": updates}}

def _planner_router(state: OrchestratorState) -> str:
    instr = state.get("instructions")
    if not instr:
        return "tool_executor"
    resp = (instr.response_generator_instruction or "").strip()
    if resp.startswith("[PROMOTION]"):
        return "slot_extractor"
    if instr.t2s_instruction or instr.knowledge_instruction:
        return "tool_executor"
    return "response_generator"

def planner_node(state: OrchestratorState):
    logger.info("--- 1. 🤔 계획 수립 노드 실행 ---")

    parser = PydanticOutputParser(pydantic_object=OrchestratorInstruction)
    history_summary = summarize_history(state.get("history", []))
    active_task_dump = state['active_task'].model_dump_json() if state.get('active_task') else 'null'
    schema_sig = state.get("schema_info", "")
    today = today_kr()

    prompt_template = textwrap.dedent("""
    You are the orchestrator for a marketing agent. Decide what to do this turn using ONLY the provided context.
    You MUST output a JSON that strictly follows: {format_instructions}

    ## Route decision (VERY IMPORTANT)
    - Decide the user's intent as one of:
      - Promotion flow (create/continue a promotion)
      - One-off answer (DB facts via t2s, or knowledge snippet)
      - Out-of-scope guidance
    - If and only if it is **promotion flow**, prefix `response_generator_instruction` with "[PROMOTION]".
    - If it is **out-of-scope guidance**, prefix with "[OUT_OF_SCOPE]".
    - Otherwise (one-off answer), no prefix.

    ## Tools
    - t2s: use for DB aggregations/rankings/trends.
    - knowledge: use for external trend summaries. Do not invent DB facts.

    ## Time normalization
    - Convert relative dates to ABSOLUTE ranges with Asia/Seoul timezone. Today is {today}.

    ## DB schema signature (hint only):
    {schema_sig}

    ## Conversation summary (last turns):
    {history_summary}

    ## Active task snapshot (JSON or null):
    {active_task}

    ## Decision rules
    - Promotion flow: do NOT call tools this turn. Just set `response_generator_instruction` (with [PROMOTION]).
    - One-off answers: set `t2s_instruction` and/or `knowledge_instruction` as needed.
    - Out-of-scope: both tools null, and provide short polite guidance with [OUT_OF_SCOPE].
    - Output must be concise, Korean polite style.

    User Message: "{user_message}"
    """)

    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
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
    })

    return {"instructions": instructions}

def action_state_node(state: OrchestratorState):
    logger.info("--- 3. 📋 액션 상태 확인 노드 실행 ---")
    slots = state.get("active_task").slots if state.get("active_task") and state.get("active_task").slots else None
    decision = get_action_state(slots=slots)
    logger.info("Action decision: %s", decision)
    return {"tool_results": {"action": decision}}

def _action_router(state: OrchestratorState) -> str:
    """
    action_state 결과로 다음 노드 결정:
    - brand/target을 묻는 상황이면 options_generator
    - 그 외 ask_for_slots/objective/duration은 response_generator
    - start_promotion/skip도 response_generator로 (LLM이 요약/안내)
    """
    tr = state.get("tool_results") or {}
    action = tr.get("action") or {}
    status = action.get("status")
    missing = action.get("missing_slots", [])

    if status == "ask_for_slots" and any(m in ("brand", "target") for m in missing):
        return "options_generator"
    return "response_generator"

def _build_candidate_t2s_instruction(target_type: str, lookback_days: int = 60) -> str:
    end = datetime.now(ZoneInfo("Asia/Seoul")).date()
    start = end - timedelta(days=lookback_days)
    # 표준 alias 강제
    if target_type == "brand_target":
        return textwrap.dedent(f"""
        최근 기간 {start}~{end}와 직전 동일 기간을 비교하여 브랜드 레벨 후보 목록을 산출해 주세요.
        반드시 다음 컬럼 alias를 포함해야 합니다:
        - brand_name
        - revenue (최근 기간 매출)
        - growth_pct (이전 동일기간 대비 증감율, %)
        - gm (최근 기간 총이익률, 0~1)
        - conversion_rate
        - repeat_rate
        - aov
        - inventory_days
        - return_rate
        - category_name
        - price_band
        - gender_age
        행은 브랜드별 1행입니다. 최근 기간 매출 상위 100개 내에서 반환해 주세요.
        """).strip()
    else:
        # category/product 관점
        return textwrap.dedent(f"""
        최근 기간 {start}~{end}와 직전 동일 기간을 비교하여 카테고리/상품 레벨 후보 목록을 산출해 주세요.
        가능한 경우 다음 컬럼 alias를 포함하세요:
        - product_id
        - product_name
        - category_name
        - revenue
        - growth_pct
        - gm
        - conversion_rate
        - repeat_rate
        - aov
        - inventory_days
        - return_rate
        - price_band
        - gender_age
        행은 상품(또는 카테고리)별 1행입니다. 최근 기간 매출 상위 200개 내에서 반환해 주세요.
        """).strip()

def options_generator_node(state: OrchestratorState):
    logger.info("--- 4. 🧠 옵션 제안 노드 실행 ---")
    thread_id = state["thread_id"]
    slots = state.get("active_task").slots if state.get("active_task") and state.get("active_task").slots else PromotionSlots()
    target_type = slots.target_type or "brand_target"

    # 1) t2s로 후보 집계
    t2s_instr = _build_candidate_t2s_instruction(target_type)
    table = run_t2s_agent_with_instruction(state["conn_str"], state["schema_info"], t2s_instr)
    rows = table["rows"]

    if not rows:
        logger.warning("t2s 후보 데이터가 비어 있습니다.")
        update_state(thread_id, {"product_options": []})
        tr = state.get("tool_results") or {}
        tr["option_candidates"] = {"candidates": [], "method": "deterministic_v1", "time_window": "", "constraints": {}}
        return {"tool_results": tr}

    # 2) knowledge 스냅샷
    knowledge = get_knowledge_snapshot()
    trending_terms = knowledge.get("trending_terms", [])

    # 3) 기회점수 계산 + 다양성 선택
    enriched = compute_opportunity_score(rows, trending_terms)
    topk = pick_diverse_top_k(enriched, k=4)

    # 4) 후보 JSON + 상태 저장
    # 라벨 구성: target_type별로 다르게
    labels: List[str] = []
    candidates: List[Dict[str, Any]] = []
    for r in topk:
        if target_type == "brand_target":
            cid = f"brand:{r.get('brand_name')}"
            label = str(r.get("brand_name") or "알 수 없는 브랜드")
            typ = "brand"
        else:
            name = r.get("product_name") or r.get("category_name") or "알 수 없는 항목"
            cid = f"product:{r.get('product_id') or name}"
            label = str(name)
            typ = "product" if r.get("product_name") else "category"
        labels.append(label)
        candidates.append({
            "id": cid,
            "label": label,
            "type": typ,
            "metrics": {k: r.get(k) for k in (
                "revenue","growth_pct","gm","conversion_rate","repeat_rate","aov","inventory_days","seasonality_score","return_rate"
            ) if k in r},
            "opportunity_score": r.get("opportunity_score"),
            "reasons": r.get("reasons", []),
            "diversity_tags": [x for x in (r.get("category_name"), r.get("price_band"), r.get("gender_age")) if x],
        })

    option_json = {
        "candidates": candidates,
        "method": "deterministic_v1",
        "time_window": "",
        "constraints": {"min_gm": 0.25, "max_return_rate": 0.1},
    }

    # Mongo: 라벨 저장
    try:
        update_state(thread_id, {"product_options": labels})
    except Exception as e:
        logger.error("옵션 라벨 저장 실패: %s", e)

    # State에도 반영
    merged_slots = _merge_slots(state, {"product_options": labels})
    logger.info("옵션 라벨 state 반영: %s", merged_slots.product_options)

    # tool_results 저장
    tr = state.get("tool_results") or {}
    tr["option_candidates"] = option_json
    return {"tool_results": tr}

def tool_executor_node(state: OrchestratorState):
    logger.info("--- 5. 🔨 툴 실행 노드 ---")
    instructions = state.get("instructions")
    existing_results = dict(state.get("tool_results") or {})
    if not instructions or (not instructions.t2s_instruction and not instructions.knowledge_instruction):
        logger.info("호출할 툴이 없습니다.")
        return {"tool_results": existing_results or None}

    tool_results: Dict[str, Any] = {}
    with ThreadPoolExecutor() as executor:
        futures = {}
        if instructions.t2s_instruction:
            futures[executor.submit(run_t2s_agent_with_instruction, state["conn_str"], state["schema_info"], instructions.t2s_instruction)] = "t2s"
        if instructions.knowledge_instruction:
            # knowledge는 간단 스텁: 문자열/요약 정도만
            futures[executor.submit(lambda: "최근 숏폼 콘텐츠를 활용한 바이럴 마케팅이 인기입니다.")] = "knowledge"

        for future in futures:
            tool_name = list(futures.values())[list(futures.keys()).index(future)]
            try:
                tool_results[tool_name] = future.result()
            except Exception as e:
                logger.error("%s 툴 실행 중 에러: %s", tool_name, e)
                tool_results[tool_name] = {"error": str(e)}

    merged = {**existing_results, **tool_results} if existing_results else tool_results
    return {"tool_results": merged}

def response_generator_node(state: OrchestratorState):
    logger.info("--- 6. 🗣️ 응답 생성 노드 ---")
    instructions = state.get("instructions")
    tr = state.get("tool_results") or {}

    instructions_text = (
        instructions.response_generator_instruction
        if instructions and instructions.response_generator_instruction
        else "사용자 요청에 맞춰 정중하고 간결하게 답변해 주세요."
    )

    t2s_table = tr.get("t2s") if isinstance(tr.get("t2s"), dict) else None
    action_decision = tr.get("action")
    knowledge_snippet = tr.get("knowledge") if isinstance(tr.get("knowledge"), str) else None
    option_candidates = tr.get("option_candidates") if isinstance(tr.get("option_candidates"), dict) else None

    prompt_tmpl = textwrap.dedent("""
    당신은 마케팅 오케스트레이터의 최종 응답 생성기입니다.
    아래 입력만을 근거로 **한국어 존댓말**로 한 번에 완성된 답변을 작성해 주세요.
    내부 도구명이나 시스템 세부 구현은 언급하지 않습니다.

    [입력 설명]
    - instructions_text: 이번 턴의 톤/방향.
    - action_decision: 프로모션 의사결정(JSON). status에 따라 필요한 문장을 작성하세요.
    - t2s_table: DB 질의 결과(JSON). 있으면 상위 10행만 마크다운 표로 미리보기.
    - knowledge_snippet: 외부 트렌드 요약(선택).
    - option_candidates: 유저에게 제안할 후보 목록(JSON). 있으면 **번호를 매겨** 제시하고, 각 항목당 1~2줄의 근거를 덧붙이세요.
      마지막에 “기타(직접 입력)” 항목도 포함해 주세요.

    [작성 지침]
    1) instructions_text를 우선합니다.
    2) action_decision.status:
       - "ask_for_slots": ask_prompts를 자연스럽게 묻습니다.
       - "start_promotion": payload 핵심만 짧게 확인합니다.
       - "skip"/없음: 무시합니다.
    3) option_candidates가 있으면:
       - 1) 2) 3) ... 처럼 **선택지를 번호로** 제시하고, 각 항목 옆에 이유(근거 지표) 1~2줄.
       - “기타(직접 입력)”도 마지막에 넣으세요.
       - 유저가 번호나 라벨로 답해도 된다고 안내하세요.
    4) t2s_table이 있으면 상위 10행 미리보기 표를 포함하되, 없는 사실은 창작하지 마세요.
    5) knowledge_snippet이 있으면 “참고 트렌드”로 1~2줄 덧붙이세요.
    6) 간결하고 구조화된 형식을 유지하세요.

    [입력 데이터]
    - instructions_text:
    {instructions_text}

    - action_decision (JSON):
    {action_decision_json}

    - t2s_table (JSON):
    {t2s_table_json}

    - option_candidates (JSON):
    {option_candidates_json}

    - knowledge_snippet:
    {knowledge_snippet}
    """)

    to_json = lambda x: json.dumps(x, ensure_ascii=False) if x is not None else "null"

    prompt = ChatPromptTemplate.from_template(prompt_tmpl)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=settings.GOOGLE_API_KEY
    )

    final_text = (prompt | llm).invoke({
        "instructions_text": instructions_text,
        "action_decision_json": to_json(action_decision),
        "t2s_table_json": to_json(ensure_table_payload(t2s_table) if t2s_table else None),
        "option_candidates_json": to_json(option_candidates),
        "knowledge_snippet": knowledge_snippet or ""
    })

    final_response = getattr(final_text, "content", None) or str(final_text)
    logger.info(f"최종 결과(L):\n{final_response}")

    history = state.get("history", [])
    history.append({"role": "user", "content": state.get("user_message", "")})
    history.append({"role": "assistant", "content": final_response})

    return {"history": history, "user_message": "", "output": final_response}

# ===== Graph =====
workflow = StateGraph(OrchestratorState)

workflow.add_node("planner", planner_node)
workflow.add_node("slot_extractor", slot_extractor_node)
workflow.add_node("action_state", action_state_node)
workflow.add_node("options_generator", options_generator_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("response_generator", response_generator_node)

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    _planner_router,
    {
        "slot_extractor": "slot_extractor",
        "tool_executor": "tool_executor",
        "response_generator": "response_generator",
    },
)

workflow.add_edge("slot_extractor", "action_state")
workflow.add_conditional_edges(
    "action_state",
    _action_router,
    {
        "options_generator": "options_generator",
        "response_generator": "response_generator",
    },
)
workflow.add_edge("options_generator", "response_generator")

workflow.add_edge("tool_executor", "response_generator")
workflow.add_edge("response_generator", END)

orchestrator_app = workflow.compile()