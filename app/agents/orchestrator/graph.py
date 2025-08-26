from __future__ import annotations

import json
import textwrap
import logging
from typing import List, Optional, Dict, Any, Literal, TypedDict, Union
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

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

# ===== Nodes =====
def _generate_llm_recommendations(state: OrchestratorState, rows: List[Dict[str, Any]], knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
    """LLM을 사용하여 DB 데이터와 지식 스냅샷을 기반으로 5개 추천 생성"""
    
    slots = state.get("active_task").slots if state.get("active_task") and state.get("active_task").slots else PromotionSlots()
    
    # 컨텍스트 정보 준비
    promotion_context = {
        "target_type": getattr(slots, 'target_type', None) or "미정",
        "brand": getattr(slots, 'brand', None) or "없음",
        "objective": getattr(slots, 'objective', None) or "미정", 
        "duration": getattr(slots, 'duration', None) or "미정",
        "budget": getattr(slots, 'budget', None) or "미정"
    }
    
    # 상위 20개 정도만 LLM에 전달 (토큰 제한 고려)
    top_rows = sorted(rows, key=lambda x: x.get('revenue', 0), reverse=True)[:20]
    
    prompt = f"""당신은 마케팅 전문가입니다. 현재 프로모션 기획 과정에서 사용자에게 제시할 상위 5개 추천 옵션을 선별하고 각각의 상세한 근거를 제공해야 합니다.

**현재 프로모션 기획 상황:**
- 타겟 유형: {promotion_context['target_type']}
- 지정 브랜드: {promotion_context['brand']}  
- 목표: {promotion_context['objective']}
- 기간: {promotion_context['duration']}
- 예산: {promotion_context['budget']}

**내부 데이터베이스 분석 결과 (상위 20개):**
{json.dumps(top_rows, ensure_ascii=False, indent=2)}

**시장 트렌드 분석:**
- 트렌딩 키워드: {knowledge.get('trending_terms', [])}
- 계절성 스파이크: {knowledge.get('seasonal_spikes', [])}
- 수집 소스: {knowledge.get('notes', [])}

**요청사항:**
1. 위 데이터를 종합 분석하여 상위 5개 추천을 선별하세요
2. 각 추천마다 다음 형태로 상세한 근거를 1-3개 제시하세요:
   - "내부 데이터베이스 분석 결과..."로 시작하는 DB 근거 (해당시)
   - "시장 트렌드 분석 결과..."로 시작하는 외부 트렌드 근거 (해당시)
   - 현재 프로모션 목표와의 연관성 (해당시)
3. 비즈니스 관점에서 마케터가 이해하기 쉽게 설명하세요

다음 JSON 형태로 정확히 응답해주세요:
{{
  "recommendations": [
    {{
      "rank": 1,
      "name": "상품/브랜드명",
      "type": "brand" 또는 "product" 또는 "category",
      "id": "원본 데이터의 식별자",
      "reasons": [
        "내부 데이터베이스 분석 결과 구체적인 근거1",
        "시장 트렌드 분석 결과 구체적인 근거2",
        "추가 비즈니스 근거3"
      ],
      "metrics_summary": "주요 성과 지표 요약"
    }}
  ]
}}"""

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers.json import JsonOutputParser
    
    logger.info("🤖 LLM 기반 추천 생성 시작...")
    logger.info("📊 입력 데이터: %d개 행, 트렌딩 용어: %s", len(top_rows), knowledge.get('trending_terms', []))
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            model_kwargs={"response_format": {"type": "json_object"}},
            api_key=settings.GOOGLE_API_KEY
        )
        
        response = llm.invoke(prompt)
        logger.info("✅ LLM 응답 수신 완료")
        
        # JSON 파싱
        try:
            result = json.loads(response.content)
            recommendations = result.get("recommendations", [])
            
            logger.info("📋 LLM 추천 결과:")
            for i, rec in enumerate(recommendations):
                logger.info("  %d. %s (%s)", i+1, rec.get("name"), rec.get("type"))
                logger.info("     근거: %s", rec.get("reasons", [])[:2])
            
            return recommendations
            
        except json.JSONDecodeError as e:
            logger.error("❌ LLM 응답 JSON 파싱 실패: %s", e)
            logger.error("응답 내용: %s", response.content[:500])
            return []
            
    except Exception as e:
        logger.error("❌ LLM 호출 실패: %s", e)
        return []

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
    - target_type은 "brand" 또는 "category" 중 하나로만.
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
        return "tool_executor"
        
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

    ## Tools
    - 사용자의 질문에 답하기 위해 필요한 모든 도구를 **tool_calls JSON 배열**에 담아 요청하세요.
    - 필요하다면 **여러 개의 도구를 하나의 배열에 동시에 요청**할 수 있습니다.
    - 각 도구 객체는 `{{"tool": "도구명", "args": {{"파라미터명": "값"}}}}` 형식을 따라야 합니다.
    - 사용 가능한 도구 목록과 형식:
      - DB 조회: `{{"tool": "t2s", "args": {{"instruction": "SQL로 변환할 자연어 질문", "output_type": "export|visualize|table"}}}}`
        - output_type 선택 가이드라인:
          * "export": 데이터를 파일로 다운로드해야 하는 경우 (예시: "클릭율이 감소 중인 유저 ID 목록", "이 데이터를 파일로 저장해줘", "리스트를 다운로드하고 싶어")
          * "visualize": 데이터 시각화가 필요한 경우 (예시: "비교"를 해야하는 질문, "상위 10개 브랜드 알려줘", "추세"에 대한 질문, "시각화해서 보여줘", "차트로 분석해줘", "그래프로 비교해줘")
          * "table": 단순 팩트 확인이나 표 형태로 보기 원하는 경우 (예: "작년 매출이 얼마였지?", "데이터를 표로 보여줘")
      - 웹 검색: `{{"tool": "tavily_search", "args": {{"query": "검색어", "max_results": 5}}}}`
      - 웹 스크래핑: `{{"tool": "scrape_webpages", "args": {{"urls": ["https://...", ...]}}}}`
      - 마케팅 트렌드: `{{"tool": "marketing_trend_search", "args": {{"question": "질문"}}}}`
      - 뷰티 트렌드: `{{"tool": "beauty_youtuber_trend_search", "args": {{"question": "질문"}}}}`
    - 도구 사용이 필요 없으면 `tool_calls` 필드를 null로 두세요.

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
    - One-off answers: set `tool_calls` as needed.
    - Out-of-scope: both tools null, and provide short polite guidance with [OUT_OF_SCOPE].
    - Output must be concise, Korean polite style.
    
    ## t2s output_type 선택 예시
    - "export" 선택 시나리오:
      * "유저 ID 목록을 엑셀로 내려줘" → output_type: "export"
      * "이 데이터를 파일로 저장해줘" → output_type: "export"  
      * "리스트를 다운로드하고 싶어" → output_type: "export"
      * "전체 데이터를 파일로 받고 싶어" → output_type: "export"
    - "visualize" 선택 시나리오:
      * "추세를 그래프로 보여줘" → output_type: "visualize"
      * "시각화해서 보여줘" → output_type: "visualize"
      * "차트로 분석해줘" → output_type: "visualize"
      * "그래프로 비교해줘" → output_type: "visualize"
      * "트렌드를 시각적으로 보여줘" → output_type: "visualize"
    - "table" 선택 시나리오:
      * "작년 매출이 얼마였지?" → output_type: "table"
      * "상위 10개 브랜드 알려줘" → output_type: "table"
      * "데이터를 표로 보여줘" → output_type: "table"
      * "매출 순위를 알려줘" → output_type: "table"
      * "어떤 브랜드가 제일 잘 팔렸어?" → output_type: "table"

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
    """
    action_state 결과로 다음 노드 결정:
    - brand/target을 묻거나, 특정 product를 물어야 하는 상황이면 options_generator로 분기
    - 그 외 (objective/duration 질문, 최종 확인 등)는 response_generator로 분기
    """
    tr = state.get("tool_results") or {}
    action = tr.get("action") or {}
    status = action.get("status")
    missing = action.get("missing_slots", [])

    # --- 👇 여기가 핵심적인 변경 부분입니다 ---
    # 'ask_for_product' 상태일 때 options_generator를 호출하도록 명시
    if status == "ask_for_product":
        return "options_generator"
    # ------------------------------------
    
    # 기존 로직: brand나 target을 물어야 할 때도 options_generator 호출
    if status == "ask_for_slots" and any(m in ("brand", "target") for m in missing):
        return "options_generator"
        
    return "response_generator"
    
def _build_candidate_t2s_instruction(target_type: str, slots: PromotionSlots) -> str:
    end = datetime.now(ZoneInfo("Asia/Seoul")).date()
    start = end - timedelta(days=60)
    
    # --- 👇 여기가 핵심적인 변경 부분입니다 ---
    # 브랜드 필터링 조건을 담을 변수
    brand_filter_instruction = ""
    # slots에 brand 정보가 있으면 필터링 지시문을 생성합니다.
    if slots and slots.brand:
        brand_filter_instruction = f" 또한, 결과는 반드시 '{slots.brand}' 브랜드의 제품만 포함해야 합니다."
    # ------------------------------------
 
    if target_type == "brand":
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
    logger.info("--- 🧠 옵션 제안 노드 실행 시작 ---")
    logger.info("📊 입력 상태 정보:")
    logger.info("  - chat_id: %s", state.get("chat_id"))
    logger.info("  - active_task 존재: %s", bool(state.get("active_task")))
    
    chat_id = state["chat_id"]
    slots = state.get("active_task").slots if state.get("active_task") and state.get("active_task").slots else PromotionSlots()
    target_type = slots.target_type or "brand"
    
    logger.info("🎯 타겟 타입: %s", target_type)
    logger.info("📋 슬롯 정보: %s", slots)

    logger.info("🔧 T2S 인스트럭션 생성 중...")
    t2s_instr = _build_candidate_t2s_instruction(target_type, slots)
    logger.info("📝 생성된 T2S 인스트럭션: %s", t2s_instr[:200] + "..." if len(t2s_instr) > 200 else t2s_instr)
    
    logger.info("🚀 T2S 에이전트 실행 중...")
    table = run_t2s_agent_with_instruction(state, t2s_instr, "visualize")  # 옵션 생성은 항상 시각화 포함
    rows = table["rows"]
    
    logger.info("📊 T2S 결과 분석:")
    logger.info("  - 전체 행 수: %d", len(rows))
    logger.info("  - 컬럼 정보: %s", list(table.get("columns", [])))
    
    if rows:
        logger.info("📋 첫 번째 행 샘플: %s", {k: v for k, v in rows[0].items() if k in ['brand_name', 'product_name', 'category_name', 'revenue', 'growth_pct']})

    if not rows:
        logger.warning("❌ T2S 후보 데이터가 비어 있습니다.")
        logger.info("🔄 빈 결과로 상태 업데이트 중...")
        update_state(chat_id, {"product_options": []})
        tr = state.get("tool_results") or {}
        tr["option_candidates"] = {"candidates": [], "method": "deterministic_v1", "time_window": "", "constraints": {}}
        logger.info("✅ 빈 옵션 후보 반환 완료")
        return {"tool_results": tr}

    logger.info("🔍 지식 스냅샷 수집 중...")
    knowledge = get_knowledge_snapshot()
    trending_terms = knowledge.get("trending_terms", [])
    
    logger.info("📈 트렌딩 용어 분석:")
    logger.info("  - 트렌딩 용어 수: %d", len(trending_terms))
    logger.info("  - 트렌딩 용어 목록: %s", trending_terms)
    logger.info("  - 계절성 스파이크: %s", knowledge.get("seasonal_spikes", []))
    logger.info("  - 수집 노트: %s", knowledge.get("notes", []))

    logger.info("🤖 LLM 기반 추천 생성 중...")
    llm_recommendations = _generate_llm_recommendations(state, rows, knowledge)
    
    if not llm_recommendations:
        logger.warning("❌ LLM 추천 생성 실패 - 기존 방식으로 폴백")
        # 폴백: 기존 방식 사용
        enriched = compute_opportunity_score(rows, trending_terms)
        topk = pick_diverse_top_k(enriched, k=5)
        
        labels = []
        candidates = []
        for i, r in enumerate(topk):
            if target_type == "brand":
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
                "metrics": {k: r.get(k) for k in ("revenue","growth_pct","gm","conversion_rate","repeat_rate","aov","inventory_days","seasonality_score","return_rate") if k in r},
                "opportunity_score": r.get("opportunity_score"),
                "reasons": r.get("reasons", []),
                "diversity_tags": [x for x in (r.get("category_name"), r.get("price_band"), r.get("gender_age")) if x],
            })
    else:
        logger.info("✅ LLM 추천 생성 성공 - %d개 추천", len(llm_recommendations))
        
        labels = []
        candidates = []
        
        for i, rec in enumerate(llm_recommendations[:5]):  # 최대 5개
            # LLM 추천을 표준 후보 형태로 변환
            name = rec.get("name", f"추천{i+1}")
            typ = rec.get("type", "product")
            
            # 원본 데이터에서 해당 항목 찾기 (메트릭 정보를 위해)
            original_row = None
            for row in rows:
                if (row.get("brand_name") == name or 
                    row.get("product_name") == name or 
                    row.get("category_name") == name):
                    original_row = row
                    break
            
            if target_type == "brand":
                cid = f"brand:{name}"
            else:
                cid = f"product:{rec.get('id', name)}"
            
            labels.append(name)
            
            candidate = {
                "id": cid,
                "label": name,
                "type": typ,
                "llm_reasons": rec.get("reasons", []),  # LLM이 생성한 상세 설명
                "metrics_summary": rec.get("metrics_summary", ""),
                "rank": rec.get("rank", i+1),
                "metrics": {k: original_row.get(k) for k in ("revenue","growth_pct","gm","conversion_rate","repeat_rate","aov","inventory_days","seasonality_score","return_rate") if original_row and k in original_row} if original_row else {},
            }
            
            candidates.append(candidate)
            
            logger.info("  %d번 LLM 추천: %s (%s)", i+1, name, typ)
            logger.info("    - 근거 개수: %d", len(rec.get("reasons", [])))
            logger.info("    - 메트릭 요약: %s", rec.get("metrics_summary", "없음")[:100])

    option_json = {
        "candidates": candidates,
        "method": "deterministic_v1",
        "time_window": "",
        "constraints": {"min_gm": 0.25, "max_return_rate": 0.1},
    }

    logger.info("💾 상태 업데이트 중...")
    try:
        update_state(chat_id, {"product_options": labels})
        logger.info("✅ 옵션 라벨 상태 저장 성공")
    except Exception as e:
        logger.error("❌ 옵션 라벨 저장 실패: %s", e)

    logger.info("🔄 슬롯 병합 중...")
    merged_slots = _merge_slots(state, {"product_options": labels})
    logger.info("✅ 옵션 라벨 state 반영: %s", merged_slots.product_options)

    logger.info("📤 최종 결과 반환 준비 중...")
    tr = state.get("tool_results") or {}
    tr["option_candidates"] = option_json
    
    logger.info("🎉 옵션 제안 노드 실행 완료!")
    logger.info("📊 최종 결과 요약:")
    logger.info("  - 후보 수: %d", len(candidates))
    logger.info("  - 라벨 목록: %s", labels)
    logger.info("  - 제약 조건: %s", option_json["constraints"])
    
    return {"tool_results": tr}

def _parse_knowledge_calls(instr: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if instr is None:
        return []

    if isinstance(instr, dict):
        return [instr]

    if isinstance(instr, list):
        # ensure list of dicts
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
        "t2s": lambda args: run_t2s_agent_with_instruction(state, args.get("instruction", ""), args.get("output_type", "table")),
        "tavily_search": lambda args: run_tavily_search(args.get("query", ""), args.get("max_results", 5)),
        "scrape_webpages": lambda args: scrape_webpages(args.get("urls", [])),
        "marketing_trend_search": lambda args: marketing_trend_search(args.get("question", "")),
        "beauty_youtuber_trend_search": lambda args: beauty_youtuber_trend_search(args.get("question", "")),
    }
    
    tool_results = {}

    with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
        future_to_call = {}
        for i, call in enumerate(tool_calls):
            tool_name = call.get("tool")
            tool_args = call.get("args", {})

            logger.info(f"🧩 {tool_name} 실행")
            
            if tool_name in tool_map:
                result_key = f"{tool_name}_{i}"
                future = executor.submit(tool_map[tool_name], tool_args)
                future_to_call[future] = result_key
            else:
                logger.warning(f"알 수 없는 도구 '{tool_name}' 호출은 건너뜁니다.")
        
        for future in future_to_call:
            result_key = future_to_call[future]

            try:
                result = future.result()
                tool_results[result_key] = result

            except Exception as e:
                logger.error(f"'{result_key}' 툴 실행 중 오류 발생: {e}")
                tool_results[result_key] = {"error": str(e)}


    existing_results = state.get("tool_results") or {}
    merged_results = {**existing_results, **tool_results}
    
    return {"tool_results": merged_results}

def _should_visualize_router(state: OrchestratorState) -> str:
    tool_results = state.get("tool_results", {})
    # tool_results 안에 t2s로 시작하고 데이터가 있는 결과가 있는지 확인
    for key, value in tool_results.items():
        if key.startswith("t2s") and value and value.get("rows"):
            output_type = value.get("output_type", "table")
            logger.info(f"T2S 결과가 있고 output_type이 '{output_type}'입니다.")
            
            # output_type에 따라 시각화 여부 결정
            if output_type == "visualize":
                logger.info("시각화를 시도합니다.")
                return "visualize"
            elif output_type == "table":
                logger.info("표만 표시하므로 시각화를 건너뜁니다.")
                return "skip_visualize"
            elif output_type == "export":
                logger.info("파일 내보내기이므로 시각화를 건너뜁니다.")
                return "skip_visualize"
            else:
                logger.info("기본값으로 표만 표시합니다.")
                return "skip_visualize"
            
    logger.info("T2S 결과가 없어 시각화를 건너뜁니다.")
    return "skip_visualize"

def visualizer_caller_node(state: OrchestratorState):
    logger.info("--- 📊 시각화 노드 실행 ---")
    
    # t2s 결과를 찾습니다. 여러 tool_results 중 t2s_0, t2s_1 등을 찾도록 수정
    t2s_result = None
    tool_results = state.get("tool_results", {})
    for key, value in tool_results.items():
        if key.startswith("t2s") and value and value.get("rows"):
            t2s_result = value
            break
    
    if not t2s_result:
        logger.info("시각화할 데이터가 없어 건너뜁니다.")
        return {}

    # Visualizer 그래프 실행
    visualizer_app = build_visualize_graph(model="gemini-2.5-flash") # 모델명은 설정에 따라 변경
    viz_state = VisualizeState(
        user_question=state.get("user_message"),
        instruction="사용자의 질문과 아래 데이터를 바탕으로 최적의 그래프를 생성하고 설명해주세요.",
        json_data=json.dumps(t2s_result, ensure_ascii=False)
    )
    
    viz_response = visualizer_app.invoke(viz_state)

    # 시각화 결과를 tool_results에 추가
    if viz_response:
        tool_results["visualization"] = {
            "json_graph": viz_response.get("json_graph"),
            "explanation": viz_response.get("output")
        }
        
    return {"tool_results": tool_results}

def response_generator_node(state: OrchestratorState):
    logger.info("--- 🗣️ 응답 생성 노드 ---")
    instructions = state.get("instructions")
    tr = state.get("tool_results") or {}

    instructions_text = (
        instructions.response_generator_instruction
        if instructions and instructions.response_generator_instruction
        else "사용자 요청에 맞춰 정중하고 간결하게 답변해 주세요."
    )

    t2s_table = None
    t2s_output_type = "table"  # 기본값
    t2s_download_url = None
    web_search = None
    scraped_pages = None
    marketing_trend_results = None
    youtuber_trend_results = None
    for key, value in tr.items():
        if key.startswith("t2s") and isinstance(value, dict) and "rows" in value:
            t2s_table = value
            t2s_output_type = value.get("output_type", "table")
            t2s_download_url = value.get("download_url")
        elif key.startswith("tavily_search"): 
            web_search = value
        elif key.startswith("scrape_webpages"):
            scraped_pages = value
        elif key.startswith("marketing_trend_search"):
            marketing_trend_results = value
        elif key.startswith("beauty_youtuber_trend_search"):
            youtuber_trend_results = value

    action_decision = tr.get("action")
    knowledge_snippet = tr.get("knowledge") if isinstance(tr.get("knowledge"), str) else None
    option_candidates = tr.get("option_candidates") if isinstance(tr.get("option_candidates"), dict) else None

    prompt_tmpl = textwrap.dedent("""
    당신은 마케팅 오케스트레이터의 최종 응답 생성기입니다.
    아래 입력만을 근거로 **한국어 존댓말**로 한 번에 완성된 답변을 작성해 주세요.
    내부 도구명이나 시스템 세부 구현은 언급하지 않습니다.

    [입력 설명]
    - instructions_text: 이번 턴의 톤/방향.
    - action_decision: 프로모션 의사결정(JSON).
    - option_candidates: 유저에게 제안할 후보 목록(JSON).
    - t2s_table: DB 질의 결과(JSON). 있으면 상위 10행만 표로 미리보기.
    - knowledge_snippet: 간단 참고(선택).
    - web_search: 웹 검색 결과(JSON: results[title,url,content]).
    - scraped_pages: 웹 페이지 본문 스크래핑 결과(JSON: documents[source,content]).
    - marketing_trend_results: Supabase 마케팅 트렌드 결과(JSON).
    - youtuber_trend_results: Supabase 뷰티 유튜버 트렌드 결과(JSON).

    [작성 지침]
    1) **가장 중요한 규칙**: `action_decision` 객체가 있고, 그 안의 `ask_prompts` 리스트에 내용이 있다면, 당신의 최우선 임무는 해당 리스트의 질문을 사용자에게 하는 것입니다. 다른 모든 지시보다 이 규칙을 **반드시** 따라야 합니다. `ask_prompts`의 문구를 그대로 사용하거나, 살짝 더 자연스럽게만 다듬어 질문하세요. (예: "타겟 종류를 선택해 주세요. (brand | category)")
    2) 위 1번 규칙에 해당하지 않는 경우에만, `instructions_text`를 주된 내용으로 삼아 답변을 생성합니다.
    3) `option_candidates`가 있으면 번호로 제시하고 각 2~4줄 근거를 붙입니다. 
       - 후보에 `llm_reasons` 필드가 있으면 그것을 우선 사용하세요 (LLM이 생성한 상세 근거)
       - `llm_reasons`가 없으면 기존 `reasons`, `business_reasons` 등을 사용하세요
       - 모든 수치는 어떤 수치인지 구체적인 언급을 해주세요
       - 마지막에 '기타(직접 입력)'도 추가합니다    
    4) web_search / scraped_pages / supabase 결과가 있으면, 핵심 근거를 2~4줄로 요약해 설명에 녹여 주세요. 원문 인용은 1~2문장 이하로 제한.
    5) t2s_table 처리 규칙:
       - output_type이 "export"인 경우: 표나 시각화를 포함하지 말고, 데이터 준비가 완료되었음을 안내하세요. 다운로드 링크는 시스템에서 자동으로 추가됩니다.
       - output_type이 "table"인 경우: 상위 10행 미리보기 표만 포함하되, 없는 수치는 만들지 마세요. 표를 시작하는 부분은 [TABLE_START] 표가 끝나는 부분은 [TABLE_END] 라는 텍스트를 붙여서 어디부터 어디가 테이블인지 알 수 있게 해주세요.
       - output_type이 "visualize"인 경우: 상위 10행 미리보기 표를 포함하고, 시각화 결과가 있다면 함께 제공하세요.
    6) 전체적으로 구조화된 형식을 유지하세요.

    [입력 데이터]
    - instructions_text:
    {instructions_text}

    - action_decision (JSON):
    {action_decision_json}

    - option_candidates (JSON):
    {option_candidates_json}

    - t2s_table (JSON):
    {t2s_table_json}

    - t2s_output_type:
    {t2s_output_type}



    - web_search (JSON):
    {web_search_json}

    - scraped_pages (JSON):
    {scraped_pages_json}

    - marketing_trend_results (JSON):
    {marketing_trend_results_json}

    - youtuber_trend_results (JSON):
    {youtuber_trend_results_json}

    - knowledge_snippet:
    {knowledge_snippet}
    """)

    to_json = lambda x: json.dumps(x, ensure_ascii=False) if x is not None else "null"

    prompt = ChatPromptTemplate.from_template(prompt_tmpl)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=settings.GOOGLE_API_KEY)
    # llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", temperature=0, api_key=settings.ANTHROPIC_API_KEY)
    
    final_text = (prompt | llm).invoke({
        "instructions_text": instructions_text,
        "action_decision_json": to_json(action_decision),
        "option_candidates_json": to_json(option_candidates),
        "t2s_table_json": to_json(t2s_table),
        "t2s_output_type": t2s_output_type,

        "web_search_json": to_json(web_search),
        "scraped_pages_json": to_json(scraped_pages),
        "marketing_trend_results_json": to_json(marketing_trend_results),
        "youtuber_trend_results_json": to_json(youtuber_trend_results),
        "knowledge_snippet": knowledge_snippet or "",
    })

    final_response = getattr(final_text, "content", None) or str(final_text)
    
    # export 타입일 때 다운로드 링크 추가
    if t2s_output_type == "export" and t2s_download_url:
        download_link = f"\n\n[CSV 다운로드]({t2s_download_url})"
        final_response += download_link
        logger.info(f"Export 링크 추가됨: {t2s_download_url}")
    elif t2s_output_type == "export" and not t2s_download_url:
        error_message = "\n\n⚠️ 파일 업로드 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
        final_response += error_message
        logger.warning("Export 요청이지만 다운로드 URL이 없습니다.")
    
    logger.info(f"최종 결과(L):\n{final_response}")
    history = state.get("history", [])
    history.append({"role": "user", "content": state.get("user_message", "")})
    history.append({"role": "assistant", "content": final_response})
    
    # logger.info(f"{state}")
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