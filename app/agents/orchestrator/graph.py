# app/agents/orchestrator/graph.py
from __future__ import annotations

import json
import textwrap
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo   
import time 
from typing import List, Optional, Dict, Any, Literal, TypedDict, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import timedelta

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from app.core.config import settings
from app.database.promotion_slots import update_state
from app.agents.promotion.state import get_action_state
from app.agents.visualizer.graph import build_visualize_graph
from app.agents.visualizer.state import VisualizeState
from .state import *
from .tools import *
from .helpers import *

logger = logging.getLogger(__name__)

# ===== Helper (length policy) =====
def _compute_length_hint(tr: Dict[str, Any], option_candidates: Optional[Dict[str, Any]], t2s_table: Optional[Dict[str, Any]]) -> str:
    """간단 휴리스틱: 표/옵션/외부근거 유무로 길이 추천"""
    has_table = bool(t2s_table and t2s_table.get("rows"))
    has_options = bool(option_candidates and option_candidates.get("candidates"))
    has_evidence = any(
        k in tr and tr.get(k)
        for k in ("marketing_trend_search_0", "tavily_search_0", "scrape_webpages_0", "beauty_youtuber_trend_search_0")
    )
    score = int(has_table) + int(has_options) + int(has_evidence)
    if score >= 2:
        return "long"    # 13~20문장
    if score == 1:
        return "medium"  # 7~12문장
    return "short"       # 4~6문장

# ===== P1-3: Deterministic 옵션 렌더링 =====
def _render_option_list_text(
    option_candidates: Optional[Dict[str, Any]],
    slots: Optional[PromotionSlots],
    *,
    max_items: int = 6,
) -> str:
    """
    후보 JSON이 있으면 그걸 우선 사용하고,
    없으면 slots.product_options(라벨 리스트)로 폴백합니다.
    최종 형식(텍스트)은 LLM이 수정하지 않도록 프롬프트에서 '그대로 포함'하도록 지시합니다.
    """
    lines: List[str] = []
    used = 0

    # 1) 후보 JSON 우선
    if option_candidates and isinstance(option_candidates.get("candidates"), list):
        for c in option_candidates["candidates"]:
            if used >= max_items:
                break
            label = str(c.get("label") or "").strip() or "선택지"
            reasons = c.get("reasons") or []
            # 2~4줄 근거 제한
            reasons = [str(r).strip() for r in reasons if r]
            if len(reasons) > 4:
                reasons = reasons[:4]
            # 핵심 메트릭을 괄호로 1줄 요약
            metrics = c.get("metrics") or {}
            mkeys = ["revenue","growth_pct","gm","conversion_rate","repeat_rate","aov","inventory_days","return_rate"]
            mparts = []
            for k in mkeys:
                if k in metrics and metrics[k] not in (None, ""):
                    mparts.append(f"{k}: {metrics[k]}")
            metrics_line = f" ({'; '.join(mparts)})" if mparts else ""
            used += 1
            idx = used
            lines.append(f"{idx}. {label}{metrics_line}")
            for r in reasons[:4]:
                lines.append(f"   - {r}")

    # 2) 폴백: slots.product_options
    if used == 0 and slots and slots.product_options:
        for i, label in enumerate(slots.product_options, start=1):
            if used >= max_items:
                break
            used += 1
            lines.append(f"{i}. {label}")

    # 3) 공통: 맨 끝에 '0. 기타(직접 입력)'
    lines.append("0. 기타(직접 입력)")
    return "\n".join(lines)

# ===== P1-4: 툴 오류 요약 렌더러 =====
def _render_tool_errors_text(tr: Dict[str, Any]) -> str:
    msgs: List[str] = []
    for key, val in (tr or {}).items():
        if isinstance(val, dict) and val.get("error"):
            tool = key.split("_")[0]
            err = val.get("error")
            if err == "timeout":
                msgs.append(f"{tool} 도구가 제한 시간(12초)을 초과하여 생략되었습니다.")
            else:
                msgs.append(f"{tool} 도구 실행 중 오류가 발생해 생략했습니다.")
    if not msgs:
        return ""
    return "참고: " + " ".join(msgs)

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
    logger.info("--- 🔍 슬롯 추출/저장 노드 실행 ---")
    user_message = state.get("user_message", "")
    chat_id = state["chat_id"]

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
        update_state(chat_id, updates)
        logger.info("Mongo 상태 업데이트: %s", updates)
    except Exception as e:
        logger.error("Mongo 업데이트 실패: %s", e)

    merged = _merge_slots(state, updates)
    logger.info("State 슬롯 병합: %s", merged.model_dump())
    return {"tool_results": {"slot_updates": updates}}

def _planner_router(state: OrchestratorState) -> str:
    instr = state.get("instructions")
    if not instr:
        return "response_generator"  # 기본 분기: 빈 툴 실행 방지

    resp = (instr.response_generator_instruction or "").strip()
    if resp.startswith("[PROMOTION]"):
        return "slot_extractor"

    if instr.tool_calls and len(instr.tool_calls) > 0:
        return "tool_executor"

    return "response_generator"

def planner_node(state: OrchestratorState):
    logger.info("--- 🤔 계획 수립 노드 실행 ---")

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

    ## Tools (MINIMIZE)
    - 먼저 질문으로 모호성을 해소하세요. **툴 호출은 필요한 경우에만** 최소 개수(가급적 1~2개)로 요청합니다.
    - 경량→중량 순서로 분해하세요: `tavily_search`로 URL/개요를 얻은 뒤 **정말 필요할 때만** `scrape_webpages`를 일부(top N) URL에 적용합니다.
    - 동시에 무거운 툴을 여러 개 호출하지 마세요. (가능하다면 순차 계획)
    - 사용 가능한 도구 목록과 형식:
      - DB 조회: `{{"tool": "t2s", "args": {{"instruction": "SQL로 변환할 자연어 질문"}}}}`
      - 웹 검색: `{{"tool": "tavily_search", "args": {{"query": "검색어", "max_results": 5}}}}`
      - 웹 스크래핑: `{{"tool": "scrape_webpages", "args": {{"urls": ["https://...", ...]}}}}`
      - 마케팅 트렌드: `{{"tool": "marketing_trend_search", "args": {{"question": "질문"}}}}`
      - 뷰티 트렌드: `{{"tool": "beauty_youtuber_trend_search", "args": {{"question": "질문"}}}}`
    - 프로모션 플로우인 경우 **이번 턴에는 툴을 호출하지 않습니다.**

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
    - One-off answers: set `tool_calls` as needed (MINIMIZED & DECOMPOSED).
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
    logger.info("--- 📋 액션 상태 확인 노드 실행 ---")
    slots = state.get("active_task").slots if state.get("active_task") and state.get("active_task").slots else None
    decision = get_action_state(slots=slots)
    logger.info("Action decision: %s", decision)
    return {"tool_results": {"action": decision}}

def _action_router(state: OrchestratorState) -> str:
    tr = state.get("tool_results") or {}
    action = tr.get("action") or {}
    status = action.get("status")
    missing = action.get("missing_slots", [])

    if status == "ask_for_product":
        return "options_generator"

    if status == "ask_for_slots" and any(m in ("brand", "target") for m in missing):
        return "options_generator"

    return "response_generator"

def _build_candidate_t2s_instruction(target_type: str, slots: PromotionSlots) -> str:
    end = datetime.now(ZoneInfo("Asia/Seoul")).date()
    start = end - timedelta(days=60)

    brand_filter_instruction = ""
    if slots and slots.brand:
        brand_filter_instruction = f" 또한, 결과는 반드시 '{slots.brand}' 브랜드의 제품만 포함해야 합니다."

    if target_type == "brand_target":
        return textwrap.dedent(f"""
        최근 기간 {start}~{end}와 직전 동일 기간을 비교하여 브랜드 레벨 후보 목록을 산출해 주세요.{brand_filter_instruction}
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
    logger.info("--- 🧠 옵션 제안 노드 실행 ---")
    chat_id = state["chat_id"]
    slots = state.get("active_task").slots if state.get("active_task") and state.get("active_task").slots else PromotionSlots()
    target_type = slots.target_type or "brand_target"

    t2s_instr = _build_candidate_t2s_instruction(target_type, slots)
    table = run_t2s_agent_with_instruction(state, t2s_instr)
    rows = table["rows"]

    if not rows:
        logger.warning("t2s 후보 데이터가 비어 있습니다.")
        update_state(chat_id, {"product_options": []})
        tr = state.get("tool_results") or {}
        tr["option_candidates"] = {"candidates": [], "method": "deterministic_v1", "time_window": "", "constraints": {}}
        return {"tool_results": tr}

    knowledge = get_knowledge_snapshot()
    trending_terms = knowledge.get("trending_terms", [])

    enriched = compute_opportunity_score(rows, trending_terms)
    topk = pick_diverse_top_k(enriched, k=4)

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

    try:
        update_state(chat_id, {"product_options": labels})
    except Exception as e:
        logger.error("옵션 라벨 저장 실패: %s", e)

    merged_slots = _merge_slots(state, {"product_options": labels})
    logger.info("옵션 라벨 state 반영: %s", merged_slots.product_options)

    tr = state.get("tool_results") or {}
    tr["option_candidates"] = option_json
    return {"tool_results": tr}

def _parse_knowledge_calls(instr: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if instr is None:
        return []
    if isinstance(instr, dict):
        return [instr]
    if isinstance(instr, list):
        return [c for c in instr if isinstance(c, dict)]
    if isinstance(instr, str):
        try:
            data = json.loads(instr)
            if isinstance(data, dict):
                return [data]
            if isinstance(data, list):
                return [c for c in data if isinstance(c, dict)]
            return []
        except Exception:
            return []
    return []

def tool_executor_node(state: OrchestratorState):
    logger.info("--- 🔨 툴 실행 노드 실행 ---")
    instructions = state.get("instructions")
    tool_calls = instructions.tool_calls if instructions and instructions.tool_calls else []
    if not tool_calls:
        logger.info("실행할 툴이 없습니다.")
        return {"tool_results": None}

    tool_map = {
        "t2s": lambda args: run_t2s_agent_with_instruction(state, args.get("instruction", "")),
        "tavily_search": lambda args: run_tavily_search(args.get("query", ""), args.get("max_results", 5)),
        "scrape_webpages": lambda args: scrape_webpages(args.get("urls", [])),
        "marketing_trend_search": lambda args: marketing_trend_search(args.get("question", "")),
        "beauty_youtuber_trend_search": lambda args: beauty_youtuber_trend_search(args.get("question", "")),
    }
    # ⏱ 도구별 타임아웃
    TOOL_TIMEOUTS = {
        "t2s": 60, 
        "scrape_webpages": 60,
        "tavily_search": 60,
        "marketing_trend_search": 60,
        "beauty_youtuber_trend_search": 60,
    }

    tool_results = {}
    MAX_WORKERS = 3
    with ThreadPoolExecutor(max_workers=min(len(tool_calls), MAX_WORKERS)) as executor:
        future_to_meta = {}
        for i, call in enumerate(tool_calls):
            name = call.get("tool")
            args = call.get("args", {})
            if name not in tool_map:
                logger.warning(f"알 수 없는 도구 '{name}' 호출은 건너뜁니다.")
                continue
            key = f"{name}_{i}"
            logger.info(f"🧩 {name} 실행")
            fut = executor.submit(tool_map[name], args)
            future_to_meta[fut] = (key, name, time.time())

        for fut, (key, name, start_ts) in future_to_meta.items():
            timeout = TOOL_TIMEOUTS.get(name, 12)
            try:
                result = fut.result(timeout=timeout)
                took = time.time() - start_ts
                logger.info(f"✅ '{key}' 완료 | {took:.2f}s")
                tool_results[key] = result
            except TimeoutError:
                logger.error(f"'{key}' 툴 실행 타임아웃({timeout}s)")
                tool_results[key] = {"error": "timeout", "message": f"요청 시간이 초과되었습니다({timeout}초).", "tool": name}
            except Exception as e:
                logger.error(f"'{key}' 툴 실행 중 오류: {e}")
                tool_results[key] = {"error": "runtime", "message": str(e), "tool": name}

    existing = state.get("tool_results") or {}
    return {"tool_results": {**existing, **tool_results}}

def _should_visualize_router(state: OrchestratorState) -> str:
    tool_results = state.get("tool_results", {})
    for key, value in tool_results.items():
        if key.startswith("t2s") and value and value.get("rows"):
            logger.info("T2S 결과가 있어 시각화를 시도합니다.")
            return "visualize"
    logger.info("T2S 결과가 없어 시각화를 건너뜁니다.")
    return "skip_visualize"

def visualizer_caller_node(state: OrchestratorState):
    logger.info("--- 📊 시각화 노드 실행 ---")
    t2s_result = None
    tool_results = state.get("tool_results", {})
    for key, value in tool_results.items():
        if key.startswith("t2s") and value and value.get("rows"):
            t2s_result = value
            break
    if not t2s_result:
        logger.info("시각화할 데이터가 없어 건너뜁니다.")
        return {}
    visualizer_app = build_visualize_graph(model="gemini-2.5-flash")
    viz_state = VisualizeState(
        user_question=state.get("user_message"),
        instruction="사용자의 질문과 아래 데이터를 바탕으로 최적의 그래프를 생성하고 설명해주세요.",
        json_data=json.dumps(t2s_result, ensure_ascii=False)
    )
    viz_response = visualizer_app.invoke(viz_state)
    if viz_response:
        tool_results["visualization"] = {
            "json_graph": viz_response.get("json_graph")       
             }
    return {"tool_results": tool_results}

def _render_table_md(table: Dict[str, Any], max_rows: int = 10) -> str:
    cols = table.get("columns") or []
    rows = (table.get("rows") or [])[:max_rows]

    def esc(x):
        s = "" if x is None else str(x)
        # 마크다운 파이프/개행 이스케이프
        return s.replace("|", r"\|").replace("\n", " ").replace("\r", " ")

    if not cols:
        # rows가 dict면 키를 추출
        if rows and isinstance(rows[0], dict):
            cols = list(rows[0].keys())
        else:
            return ""

    header = "| " + " | ".join(esc(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]

    for r in rows:
        if isinstance(r, dict):
            vals = [esc(r.get(c, "")) for c in cols]
        else:
            vals = [esc(v) for v in r]
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)

def response_generator_node(state: OrchestratorState):
    logger.info("--- 🗣️ 응답 생성 노드 (callback streaming via .invoke) ---")
    instructions = state.get("instructions")
    tr = state.get("tool_results") or {}

    instructions_text = (
        instructions.response_generator_instruction
        if instructions and instructions.response_generator_instruction
        else "사용자 요청에 맞춰 정중하고 간결하게 답변해 주세요."
    )

    # 툴 결과 수집
    t2s_table = None
    web_search = None
    scraped_pages = None
    marketing_trend_results = None
    youtuber_trend_results = None
    for key, value in tr.items():
        if key.startswith("t2s") and isinstance(value, dict) and "rows" in value:
            t2s_table = value
        elif key.startswith("tavily_search"):
            web_search = value
        elif key.startswith("scrape_webpages"):
            scraped_pages = value
        elif key.startswith("marketing_trend_search"):
            marketing_trend_results = value
        elif key.startswith("beauty_youtuber_trend_search"):
            youtuber_trend_results = value

    action_decision   = tr.get("action")
    knowledge_snippet = tr.get("knowledge") if isinstance(tr.get("knowledge"), str) else None
    option_candidates = tr.get("option_candidates") if isinstance(tr.get("option_candidates"), dict) else None

    # 선택지/오류 요약/길이 힌트
    slots = state.get("active_task").slots if state.get("active_task") and state.get("active_task").slots else PromotionSlots()
    option_list_text  = _render_option_list_text(option_candidates, slots)
    tool_errors_text  = _render_tool_errors_text(tr)
    length_hint       = _compute_length_hint(tr, option_candidates, t2s_table)

    prompt_tmpl = textwrap.dedent("""
    # 길이 규칙(필수) → {length_hint}
    - short=4~6문장, medium=7~12문장, long=13~20문장

    당신은 마케팅 오케스트레이터의 최종 응답 생성기입니다.
    아래 입력만을 근거로 **한국어 존댓말**로 한 번에 완성된 답변을 작성하세요.
    내부 도구명/시스템 구현은 언급하지 마세요.

    [작성 우선순위]
    1) action_decision.ask_prompts가 있으면 가장 먼저 공손한 한 문장 질문.
    2) 다음 줄에 **option_list_text**를 변형 없이 그대로 출력.
    3) 근거/설명은 2~4줄(수치가 있으면 구체적으로 표기).
    4) (주의) [TABLE_START]/[TABLE_END] 토큰은 **서버가 삽입**합니다. 모델은 사용하지 마세요.
    5) 마지막 줄: 다음 단계 1문장.
    6) tool_errors_text가 있으면 맨 끝 한 줄에만 첨부(“참고: …”).

    - instructions_text:
    {instructions_text}

    - action_decision (JSON):
    {action_decision_json}

    - option_list_text (그대로 출력):
    {option_list_text}

    - t2s_table (요약 JSON):
    {t2s_table_json}

    - web_search (JSON):
    {web_search_json}

    - scraped_pages (JSON):
    {scraped_pages_json}

    - marketing_trend_results (JSON):
    {marketing_trend_results_json}

    - youtuber_trend_results (JSON):
    {youtuber_trend_results_json}

    - tool_errors_text:
    {tool_errors_text}

    - knowledge_snippet:
    {knowledge_snippet}
    """)

    to_json = lambda x: json.dumps(x, ensure_ascii=False) if x is not None else "null"

    prompt = ChatPromptTemplate.from_template(prompt_tmpl)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=settings.GOOGLE_API_KEY,
    )
    chain = prompt | llm

    inputs = {
        "length_hint": length_hint,
        "instructions_text": instructions_text,
        "action_decision_json": to_json(action_decision),
        "option_list_text": option_list_text,
        "t2s_table_json": to_json({
            "columns": (t2s_table or {}).get("columns"),
            "row_count": (t2s_table or {}).get("row_count")
        }),
        "web_search_json": to_json(web_search),
        "scraped_pages_json": to_json(scraped_pages),
        "marketing_trend_results_json": to_json(marketing_trend_results),
        "youtuber_trend_results_json": to_json(youtuber_trend_results),
        "tool_errors_text": tool_errors_text or "",
        "knowledge_snippet": knowledge_snippet or "",
    }

    res = chain.invoke(inputs)
    final_response = getattr(res, "content", None) or str(res) or ""

    table_md = ""
    if t2s_table and t2s_table.get("rows"):
        table_md = _render_table_md(t2s_table, max_rows=10)
    if table_md:
        final_response = f"{final_response}\n\n[TABLE_START]\n{table_md}\n[TABLE_END]"

    logger.info(f"최종 결과(callback stream):\n{final_response}")

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
workflow.add_node("visualizer", visualizer_caller_node)
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
workflow.add_conditional_edges(
    "tool_executor",
    _should_visualize_router,
    {
        "visualize": "visualizer",
        "skip_visualize": "response_generator"
    }
)
workflow.add_edge("visualizer", "response_generator")
workflow.add_edge("response_generator", END)

orchestrator_app = workflow.compile()
