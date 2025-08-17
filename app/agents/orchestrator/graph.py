# app/agents/orchestrator/graph.py
from __future__ import annotations

import json
import textwrap
import logging
import re
from typing import List, Optional, Dict, Any, Literal, TypedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from app.agents.text_to_sql.__init__ import call_sql_generator
from .state import *

logger = logging.getLogger(__name__)


# =========================
# Tools
# =========================

def run_t2s_agent(state: OrchestratorState):
    instruction = state['instructions'].t2s_instruction
    logger.info("T2S 에이전트 실행: %s", instruction)

    result = call_sql_generator(message=instruction, conn_str=state['conn_str'], schema_info=state['schema_info'])
    sql = result['query']
    table = result["data_json"]

    logger.info(f"쿼리: \n{sql}")
    logger.info(f"결과 테이블: \n{table}")

    if isinstance(table, str):
        table = json.loads(table)
    return table

def run_knowledge_agent(instruction: str) -> str:
    logger.info("지식 에이전트 실행: %s", instruction)
    return "최근 숏폼 콘텐츠를 활용한 바이럴 마케팅이 인기입니다."

# =========================
# Helpers
# =========================

def _summarize_history(history: List[Dict[str, str]], limit_chars: int = 800) -> str:
    """최근 히스토리를 간단 요약으로 제공 (LLM 컨텍스트용)"""
    text = " ".join(h.get("content", "") for h in history[-6:])
    return text[:limit_chars]

def _today_kr() -> str:
    """Asia/Seoul 기준 오늘 날짜 yyyy-mm-dd"""
    try:
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

def _normalize_table(table: Any) -> Dict[str, Any]:
    """
    t2s 표 결과를 표준 형태로 정규화.

    지원하는 입력:
      - {"rows":[...], "columns":[...]}                           # 이미 표준
      - {"columns":[...], "data":[[...], ...]}                    # pandas orient='split'
      - {"schema": {...}, "data":[{...}, ...]}                    # pandas orient='table'
      - [{"col": val, ...}, ...]                                  # pandas orient='records'
      - {col: {row_idx: val, ...}, ...}                           # pandas orient='columns'
      - {row_idx: {col: val, ...}, ...}                           # pandas orient='index'
      - [[...], [...]]                                            # 열 이름 미상 (col_0.. 생성)
    """
    # 문자열이면 JSON 먼저 파싱
    if isinstance(table, str):
        try:
            table = json.loads(table)
        except Exception:
            return {"rows": [], "columns": [], "row_count": 0}

    # 0) 이미 표준
    if isinstance(table, dict) and "rows" in table:
        rows = table.get("rows") or []
        cols = table.get("columns") or (list(rows[0].keys()) if rows and isinstance(rows[0], dict) else [])
        return {"rows": rows, "columns": cols, "row_count": len(rows)}

    # 1) split
    if isinstance(table, dict) and "columns" in table and "data" in table and isinstance(table["data"], list):
        cols = table["columns"]
        data = table["data"]
        rows = [{cols[i]: (row[i] if i < len(row) else None) for i in range(len(cols))} for row in data]
        return {"rows": rows, "columns": cols, "row_count": len(rows)}

    # 2) table
    if isinstance(table, dict) and "schema" in table and "data" in table and isinstance(table["data"], list):
        data = table["data"]
        if data and isinstance(data[0], dict):
            cols = list(data[0].keys())
            return {"rows": data, "columns": cols, "row_count": len(data)}

    # 3) records
    if isinstance(table, list) and (not table or isinstance(table[0], dict)):
        cols = list(table[0].keys()) if table else []
        return {"rows": table, "columns": cols, "row_count": len(table)}

    # 4) columns (dict-of-dicts or dict-of-lists)
    if isinstance(table, dict) and table and all(isinstance(v, (dict, list)) for v in table.values()):
        cols = list(table.keys())
        # dict-of-dicts: {col: {row_idx: val}}
        if all(isinstance(v, dict) for v in table.values()):
            row_keys = set()
            for d in table.values():
                row_keys |= set(d.keys())

            def _ord(k):
                try: return int(k)
                except Exception:
                    try: return float(k)
                    except Exception:
                        return str(k)

            ordered = sorted(list(row_keys), key=_ord)
            rows = []
            for rk in ordered:
                row = {c: table[c].get(rk) for c in cols}
                rows.append(row)
            return {"rows": rows, "columns": cols, "row_count": len(rows)}

        # dict-of-lists: {col: [v0, v1, ...]}
        if all(isinstance(v, list) for v in table.values()):
            maxlen = max((len(v) for v in table.values()), default=0)
            rows = [{c: (table[c][i] if i < len(table[c]) else None) for c in cols} for i in range(maxlen)]
            return {"rows": rows, "columns": cols, "row_count": len(rows)}

    # 5) index orientation: {row_idx: {col: val}}
    if isinstance(table, dict) and table and all(isinstance(v, dict) for v in table.values()):
        try:
            items = list(table.items())

            def _ord2(k):
                try: return int(k)
                except Exception:
                    try: return float(k)
                    except Exception:
                        return str(k)

            items.sort(key=lambda kv: _ord2(kv[0]))
            rows = [kv[1] for kv in items]
            cols = list(rows[0].keys()) if rows else []
            return {"rows": rows, "columns": cols, "row_count": len(rows)}
        except Exception:
            pass

    # 6) list-of-lists
    if isinstance(table, list) and table and isinstance(table[0], (list, tuple)):
        max_cols = max((len(r) for r in table), default=0)
        cols = [f"col_{i}" for i in range(max_cols)]
        rows = [{cols[i]: v for i, v in enumerate(r)} for r in table]
        return {"rows": rows, "columns": cols, "row_count": len(rows)}

    return {"rows": [], "columns": [], "row_count": 0}

def _pick_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lc = [c.lower() for c in columns]
    for cand in candidates:
        if cand.lower() in lc:
            return columns[lc.index(cand.lower())]
    # 부분 일치도 허용(예: 'product_name', 'product')
    for i, c in enumerate(lc):
        for cand in candidates:
            if cand.lower() in c:
                return columns[i]
    return None

def _format_number(n: Any) -> str:
    try:
        x = float(n)
        # 정수처럼 보이면 정수로, 아니면 소수 2자리
        if abs(x - int(x)) < 1e-9:
            return f"{int(x):,}"
        return f"{x:,.2f}"
    except Exception:
        return str(n)

_NUMBER_STRIP_RE = re.compile(r"[,\s₩$€£]|(?<=\d)\%")

def _to_float_safe(v: Any) -> Optional[float]:
    """문자·통화·퍼센트 등을 안전하게 float 변환"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    # 괄호 음수 (예: (1,234))
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = _NUMBER_STRIP_RE.sub("", s)
    try:
        x = float(s)
        return -x if neg else x
    except Exception:
        return None

def _markdown_table(rows: List[Dict[str, Any]], columns: List[str], limit: int = 10) -> str:
    if not rows:
        return "_표시할 데이터가 없습니다._"
    cols = columns or list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in rows[:limit]:
        line = "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        lines.append(line)
    if len(rows) > limit:
        lines.append(f"\n_표시는 상위 {limit}행 미리보기입니다 (총 {len(rows)}행)._")
    return "\n".join(lines)

def _format_period_by_datecol(rows: List[Dict[str, Any]], date_col: Optional[str]) -> str:
    """date 열이 있으면 min~max 기간을 표시"""
    if not rows or not date_col:
        return ""
    vals = []
    for r in rows:
        v = r.get(date_col)
        if v is None:
            continue
        s = str(v)
        # 단순 파싱 (YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD 등)
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m", "%Y/%m"):
            try:
                d = datetime.strptime(s[:len(fmt)], fmt)
                vals.append(d)
                break
            except Exception:
                continue
    if not vals:
        return ""
    start = min(vals).strftime("%Y-%m-%d")
    end = max(vals).strftime("%Y-%m-%d")
    return f" (기간: {start} ~ {end})"


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
    # tool_results에 누적 병합될 수 있도록 반환
    return {"tool_results": {"action": decision}}

def tool_executor_node(state: OrchestratorState):
    logger.info("--- 3. 🔨 툴 실행 노드 (Tool Executor) 실행 ---")

    instructions = state.get("instructions")
    # 기존 결과와 병합(액션 노드 결과 유지)
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

    # 액션 결과와 병합
    merged = {**existing_results, **tool_results} if existing_results else tool_results
    return {"tool_results": merged}

def response_generator_node(state: OrchestratorState):
    logger.info("--- 4. 🗣️ 응답 생성 노드 (Response Generator) 실행 ---")

    instructions = state.get("instructions")
    tool_results = state.get("tool_results") or {}

    # LLM에 전달할 입력을 '그대로' 구성 (LLM이 서식/분기 모두 결정)
    instructions_text = (
        instructions.response_generator_instruction
        if instructions and instructions.response_generator_instruction
        else "사용자 요청에 맞춰 정중하고 간결하게 답변해 주세요."
    )

    # 표는 정규화만 해서 전달 (행 조립/정렬/숫자 파싱 등 일체 금지)
    t2s_payload = tool_results.get("t2s")
    t2s_table = _normalize_table(t2s_payload) if t2s_payload else None

    action_decision = tool_results.get("action")  # action_state_node 결과 그대로
    knowledge_snippet = tool_results.get("knowledge") if isinstance(tool_results.get("knowledge"), str) else None

    # LLM 프롬프트 (최종 응답을 LLM이 직접 작성)
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

    # JSON 직렬화(LLM이 그대로 읽도록): 가공·해석 없이 전달
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

    # 모델 객체/문자열 호환
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
