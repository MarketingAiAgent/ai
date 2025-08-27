from __future__ import annotations

import json
import textwrap
import logging
from typing import List, Optional, Dict, Any, Literal, TypedDict, Union
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, date, datetime

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
        "focus": getattr(slots, 'focus', None) or "없음",
        "target": getattr(slots, 'target', None) or "미정",
        "objective": getattr(slots, 'objective', None) or "미정", 
        "duration": getattr(slots, 'duration', None) or "미정",
    }
    
    # 상위 20개 정도만 LLM에 전달 (토큰 제한 고려) - None 값 안전 처리
    def safe_revenue_key(x):
        revenue = x.get('revenue', 0)
        # None이나 NaN 값을 0으로 처리
        if revenue is None or (isinstance(revenue, (int, float)) and revenue != revenue):  # NaN 체크
            return 0
        try:
            return float(revenue)
        except (ValueError, TypeError):
            return 0
    
    top_rows = sorted(rows, key=safe_revenue_key, reverse=True)[:20]
    
    prompt = f"""당신은 마케팅 전문가입니다. 현재 프로모션 기획 과정에서 사용자에게 제시할 상위 5개 추천 옵션을 선별하고 각각의 상세한 근거를 제공해야 합니다.

**현재 프로모션 기획 상황:**
- 타겟 유형: {promotion_context['target_type']}
- 포커스: {promotion_context['focus']}
- 타겟 고객층: {promotion_context['target']}  
- 목표: {promotion_context['objective']}
- 기간: {promotion_context['duration']}

**내부 데이터베이스 분석 결과 (상위 20개):**
{json.dumps(top_rows, ensure_ascii=False, indent=2, default=str)}

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
4. **타입 구분**: 
   - 브랜드 프로모션: "type": "brand"
   - 카테고리 프로모션 (카테고리 선택 단계): "type": "category"
   - 카테고리 프로모션 (상품 선택 단계): "type": "product"

**중요: 반드시 아래 JSON 형태로만 응답하세요. 마크다운, 설명, 코드펜스 등은 절대 포함하지 마세요.**

{{
  "recommendations": [
    {{
      "rank": 1,
      "name": "상품/브랜드/카테고리명",
      "type": "{promotion_context['target_type']}",
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
        # 프롬프트 크기 체크 (너무 크면 축소)
        if len(prompt) > 50000:  # 50KB 제한
            logger.warning("⚠️ 프롬프트가 너무 큽니다 (%d자), 데이터 축소 중...", len(prompt))
            top_rows = top_rows[:10]  # 20개에서 10개로 축소
            prompt = f"""당신은 마케팅 전문가입니다. 현재 프로모션 기획 과정에서 사용자에게 제시할 상위 5개 추천 옵션을 선별하고 각각의 상세한 근거를 제공해야 합니다.

**현재 프로모션 기획 상황:**
- 타겟 유형: {promotion_context['target_type']}
- 포커스: {promotion_context['focus']}
- 타겟 고객층: {promotion_context['target']}  
- 목표: {promotion_context['objective']}
- 기간: {promotion_context['duration']}

**내부 데이터베이스 분석 결과 (상위 10개):**
{json.dumps(top_rows, ensure_ascii=False, indent=2, default=str)}

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
4. **타입 구분**: 
   - 브랜드 프로모션: "type": "brand"
   - 카테고리 프로모션 (카테고리 선택 단계): "type": "category"
   - 카테고리 프로모션 (상품 선택 단계): "type": "product"

**중요: 반드시 아래 JSON 형태로만 응답하세요. 마크다운, 설명, 코드펜스 등은 절대 포함하지 마세요.**

{{
  "recommendations": [
    {{
      "rank": 1,
      "name": "상품/브랜드/카테고리명",
      "type": "{promotion_context['target_type']}",
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

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            model_kwargs={"response_format": {"type": "json_object"}},
            api_key=settings.GOOGLE_API_KEY
        )
        
        logger.info("📤 LLM 호출 중... (프롬프트 크기: %d자)", len(prompt))
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
            
            # JSON 파싱 실패 시 응답에서 JSON 부분만 추출 시도
            try:
                content = response.content
                # ```json과 ``` 사이의 내용 추출
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if end > start:
                        json_content = content[start:end].strip()
                        result = json.loads(json_content)
                        recommendations = result.get("recommendations", [])
                        logger.info("✅ JSON 블록에서 추출 성공: %d개 추천", len(recommendations))
                        return recommendations
            except Exception as extract_error:
                logger.error("❌ JSON 블록 추출도 실패: %s", extract_error)
            
            return []
            
    except Exception as e:
        logger.error("❌ LLM 호출 실패: %s", e)
        return []

def _merge_slots(state: OrchestratorState, updates: Dict[str, Any]) -> PromotionSlots:
    current = (state.get("active_task").slots if state.get("active_task") and state.get("active_task").slots else PromotionSlots())
    base = current.model_dump()
    for k, v in updates.items():
        if v not in (None, "", []):
            # 리스트 타입 필드들은 기존 값과 병합
            if k in ("selected_product", "product_options") and isinstance(v, list):
                existing = base.get(k, [])
                if isinstance(existing, list):
                    # 중복 제거하면서 병합
                    base[k] = list(set(existing + v))
                else:
                    base[k] = v
            else:
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
    
    **추출 규칙:**
    - 존재하는 값만 채우고, 없으면 null로 두세요.
    - 명시적으로 언급되지 않은 필드는 절대 추측하지 마세요.
    - target_type은 "brand" 또는 "category" 중 하나로만.
    - 날짜/기간은 원문 그대로 문자열로 유지.
    - focus: 사용자가 선택한 브랜드명 또는 카테고리명 (예: "나이키", "스포츠웨어")
    - target: 타겟 고객층 - 명시적으로 언급된 경우에만 (예: "20대 남성", "직장인")
    - selected_product: 사용자가 선택한 구체적인 상품명들의 리스트 (예: ["상품A", "상품B"])
    - wants_trend: 트렌드 반영 여부 (예: "예", "네", "트렌드", "좋아", "해줘" → true, "아니오", "아니", "없이", "안해", "괜찮아" → false)
    - objective: 프로모션 목표 - 명시적으로 언급된 경우에만 (예: "매출 증대", "신규 고객 유입")
    
    **금지사항:**
    - budget, cost, 예산 등은 추출하지 마세요 (슬롯에 없는 필드임)
    - 브랜드/카테고리와 상품명을 명확히 구분하세요. 브랜드/카테고리는 focus 필드에, 구체적인 상품은 selected_product에 넣으세요.
    
    **CRITICAL: 출력은 반드시 유효한 JSON 형태여야 합니다. 다른 텍스트나 설명 없이 JSON만 반환하세요.**
    
    JSON 스키마:
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
    
    # 이미 설정된 슬롯은 변경하지 않음 (기존 값 보존)
    current_slots = state.get("active_task").slots if state.get("active_task") and state.get("active_task").slots else PromotionSlots()
    
    # 이미 채워진 필드들은 업데이트에서 제거
    preserved_fields = []
    for field in ["target_type", "focus", "duration", "selected_product", "wants_trend"]:
        current_value = getattr(current_slots, field, None)
        if current_value not in (None, "", []) and field in updates:
            preserved_fields.append(field)
            logger.info("%s이(가) 이미 설정됨 (%s), 변경 방지", field, current_value)
            del updates[field]
    
    if preserved_fields:
        logger.info("보존된 필드들: %s", preserved_fields)
    
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
        logger.warning("⚠️ instructions가 None입니다. planner_node에서 에러 발생했을 가능성 높음")
        logger.warning("상태 정보: user_message='%s'", state.get('user_message', 'N/A')[:100])
        return "response_generator"
        
    resp = (instr.response_generator_instruction or "").strip()
    logger.info("라우팅 결정 중: response_instruction='%s'", resp[:50])
    
    if resp.startswith("[PROMOTION]"):
        logger.info("→ 프로모션 플로우: slot_extractor")
        return "slot_extractor"
    

    if instr.tool_calls and len(instr.tool_calls) > 0:
        logger.info("→ 툴 호출 필요: tool_executor (%d개 툴)", len(instr.tool_calls))
        return "tool_executor"
        
    logger.info("→ 응답 생성: response_generator")
    return "response_generator"

def trend_planner_node(state: OrchestratorState):
    """트렌드 반영을 위한 외부 데이터 수집 계획"""
    logger.info("--- 🌟 트렌드 수집 계획 수립 노드 실행 ---")
    
    slots = state.get("active_task").slots if state.get("active_task") and state.get("active_task").slots else PromotionSlots()
    
    # 프로모션 정보를 바탕으로 트렌드 검색 쿼리 생성
    search_queries = []
    
    # 기본 검색어 구성
    base_query = f"{slots.focus or ''} {slots.target or ''} 프로모션 마케팅"
    search_queries.append(base_query.strip())
    
    # 타겟별 트렌드 검색
    if slots.target:
        search_queries.append(f"{slots.target} 트렌드 유행어 밈")
    
    # 상품별 트렌드 검색  
    if slots.selected_product:
        for product in slots.selected_product[:2]:  # 최대 2개 상품만
            search_queries.append(f"{product} 마케팅 트렌드")
    
    # 외부 데이터 수집 툴 호출 구성
    tool_calls = [
        {"tool": "tavily_search", "args": {"query": search_queries[0], "max_results": 5}},
        {"tool": "marketing_trend_search", "args": {"question": f"{slots.focus} {slots.target} 프로모션 관련 최신 트렌드"}},
    ]
    
    # 뷰티 관련이면 유튜버 트렌드도 추가
    if any(keyword in (slots.focus or "").lower() for keyword in ["화장품", "뷰티", "코스메틱", "스킨케어"]):
        tool_calls.append({
            "tool": "beauty_youtuber_trend_search", 
            "args": {"question": f"{slots.focus} 뷰티 트렌드"}
        })
    
    return {
        "instructions": OrchestratorInstruction(
            tool_calls=tool_calls,
            response_generator_instruction="트렌드 데이터를 수집하여 프로모션에 적용할 준비를 하고 있습니다."
        )
    }

def planner_node(state: OrchestratorState):
    logger.info("--- 🤔 계획 수립 노드 실행 ---")
    logger.info("입력 상태: user_message='%s', active_task=%s", 
                state.get('user_message', 'N/A')[:100], 
                bool(state.get('active_task')))
    
    try:
        # 트렌드 반영 상태인지 확인
        tr = state.get("tool_results") or {}
        action = tr.get("action") or {}
        if action.get("status") == "apply_trends":
            logger.info("트렌드 반영 상태 감지, trend_planner_node로 전환")
            return trend_planner_node(state)

        parser = PydanticOutputParser(pydantic_object=OrchestratorInstruction)
        history_summary = summarize_history(state.get("history", []))
        active_task_dump = state['active_task'].model_dump_json() if state.get('active_task') else 'null'
        schema_sig = state.get("schema_info", "")
        today = today_kr()

        prompt_template = textwrap.dedent("""
    You are the orchestrator for a marketing agent. Decide what to do this turn using ONLY the provided context.
    
    **CRITICAL: You MUST output ONLY valid JSON. No explanations, no markdown, no code blocks - just pure JSON.**
    
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
    - **트렌드 반영 프로모션**: 다음 경우에 마케팅 트렌드 수집 툴들을 호출하세요:
      * wants_trend=true이고 필요 슬롯이 모두 채워진 경우 
      * 또는 이전 메시지가 트렌드 질문이고 현재 사용자가 긍정적으로 응답한 경우 (예: "예", "네", "응", "좋아", "해줘" 등)
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
    - **EXCEPTION 1**: If active_task shows promotion is ready for trend application (wants_trend=true, all slots filled), then call trend tools instead of setting [PROMOTION].
    - **EXCEPTION 2**: If user is responding positively to a trend question (check conversation history for recent trend question + current positive response), then call trend tools even if wants_trend is still None.
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

        # llm = ChatAnthropic(
        #     model="claude-sonnet-4-20250514", 
        #     temperature=0.1,
        #     max_tokens=8192,  # 넉넉한 토큰 제한 설정
        #     api_key=settings.ANTHROPIC_API_KEY
        # )

        logger.info("LLM 호출 중...")
        try:
            instructions = (prompt | llm | parser).invoke({
                "user_message": state['user_message'],
                "history_summary": history_summary,
                "active_task": active_task_dump,
                "schema_sig": schema_sig,
                "today": today,
            })
        except Exception as parse_error:
            logger.warning("JSON 파싱 실패, 재시도: %s", parse_error)
            # 재시도 (더 강한 프롬프트로)
            retry_prompt = prompt_template + "\n\n**REMINDER: Output must be valid JSON only!**"
            retry_template = ChatPromptTemplate.from_template(
                template=retry_prompt,
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            instructions = (retry_template | llm | parser).invoke({
                "user_message": state['user_message'],
                "history_summary": history_summary,
                "active_task": active_task_dump,
                "schema_sig": schema_sig,
                "today": today,
            })

        logger.info("✅ 계획 수립 성공: tool_calls=%s, response_instruction=%s", 
                   len(instructions.tool_calls or []), 
                   instructions.response_generator_instruction[:100] if instructions.response_generator_instruction else 'N/A')
        return {"instructions": instructions}
        
    except Exception as e:
        logger.error("❌ 계획 수립 실패: %s", e, exc_info=True)
        logger.error("에러 타입: %s", type(e).__name__)
        logger.error("사용자 메시지: %s", state.get('user_message', 'N/A'))
        logger.error("히스토리 길이: %d", len(state.get('history', [])))
        
        # 폴백 instructions 생성
        fallback_instructions = OrchestratorInstruction(
            tool_calls=None,
            response_generator_instruction="죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해 주세요."
        )
        logger.info("폴백 instructions 생성 완료")
        return {"instructions": fallback_instructions}

def action_state_node(state: OrchestratorState):
    logger.info("--- 📋 액션 상태 확인 노드 실행 ---")
    active_task = state.get("active_task")
    slots = active_task.slots if active_task and active_task.slots else None
    logger.info("입력: active_task=%s, slots=%s", bool(active_task), slots.model_dump() if slots else None)
    
    decision = get_action_state(slots=slots)
    logger.info("✅ Action decision: %s", decision)
    return {"tool_results": {"action": decision}}

def _action_router(state: OrchestratorState) -> str:
    """
    action_state 결과로 다음 노드 결정:
    - brand/target을 묻거나, 특정 product를 물어야 하는 상황이면 options_generator로 분기
    - 트렌드 반영이 필요하면 tool_executor로 분기
    - 그 외 (objective/duration 질문, 최종 확인 등)는 response_generator로 분기
    """
    tr = state.get("tool_results") or {}
    action = tr.get("action") or {}
    status = action.get("status")
    missing = action.get("missing_slots", [])

    # 트렌드 반영 상태일 때 외부 데이터 툴 호출
    if status == "apply_trends":
        return "tool_executor"
    
    # 최종 기획서 생성 상태
    if status == "create_final_plan":
        return "response_generator"

    # 'ask_for_product' 상태일 때 options_generator를 호출하도록 명시
    if status == "ask_for_product":
        return "options_generator"
    
    # 기존 로직: focus나 target을 물어야 할 때도 options_generator 호출
    if status == "ask_for_slots" and any(m in ("focus", "target") for m in missing):
        return "options_generator"
        
    return "response_generator"
    
def _build_candidate_t2s_instruction(target_type: str, slots: PromotionSlots) -> str:
    end = datetime.now(ZoneInfo("Asia/Seoul")).date()
    start = end - timedelta(days=30)  # 30일로 단축
    
    # target 조건 추가
    target_filter = ""
    if slots and slots.target:
        target_filter = f" '{slots.target}' 타겟 고객층이 주로 구매하는"
    
    if target_type == "brand":
        if slots and slots.focus:
            # 브랜드가 선택된 경우 해당 브랜드의 상품 옵션
            return f"'{slots.focus}' 브랜드의{target_filter} 최근 30일 매출 상위 20개 상품을 product_name, revenue, growth_pct 컬럼으로 조회해 주세요."
        else:
            # 브랜드 선택 단계
            return f"{target_filter} 최근 30일 매출 상위 15개 브랜드를 brand_name, revenue, growth_pct 컬럼으로 조회해 주세요."
    else:
        if slots and slots.focus:
            # 카테고리가 선택된 경우 해당 카테고리의 상품 옵션
            return f"'{slots.focus}' 카테고리의{target_filter} 최근 30일 매출 상위 20개 상품을 product_name, brand_name, revenue, growth_pct 컬럼으로 조회해 주세요."
        else:
            # 카테고리 선택 단계
            return f"{target_filter} 최근 30일 매출 상위 15개 카테고리를 category_name, revenue, growth_pct 컬럼으로 조회해 주세요."

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
        # 폴백: 단순 점수 기반 선택
        enriched = compute_opportunity_score(rows, trending_terms)
        # 다양성 제약 없이 상위 점수 순으로 선택
        topk = sorted(enriched, key=lambda x: x.get("opportunity_score", 0), reverse=True)[:5]
        
        labels = []
        candidates = []
        for i, r in enumerate(topk):
            if target_type == "brand":
                cid = f"brand:{r.get('brand_name')}"
                label = str(r.get("brand_name") or "알 수 없는 브랜드")
                typ = "brand"
            else:
                # 카테고리 타입일 때는 카테고리를 우선적으로 추천
                if r.get("category_name") and not slots.focus:
                    # 카테고리 선택 단계
                    name = r.get("category_name")
                    cid = f"category:{name}"
                    label = str(name)
                    typ = "category"
                else:
                    # 특정 카테고리가 선택된 경우 상품 추천
                    name = r.get("product_name") or r.get("category_name") or "알 수 없는 항목"
                    cid = f"product:{r.get('product_id') or name}"
                    label = str(name)
                    typ = "product" if r.get("product_name") else "category"
            
            labels.append(label)
            candidates.append({
                "id": cid,
                "label": label,
                "type": typ,
                "metrics": {k: r.get(k) for k in ("revenue","growth_pct","gm") if k in r},
                "opportunity_score": r.get("opportunity_score"),
                "reasons": r.get("reasons", []),
                "score": r.get("opportunity_score"),
            })
            
            logger.info("  %d번 폴백 추천: %s (%s)", i+1, label, typ)
    else:
        logger.info("✅ LLM 추천 생성 성공 - %d개 추천", len(llm_recommendations))
        
        labels = []
        candidates = []
        
        for i, rec in enumerate(llm_recommendations[:5]):  # 최대 5개
            # LLM 추천을 표준 후보 형태로 변환
            name = rec.get("name", f"추천{i+1}")
            typ = rec.get("type", "product")
            
            # 원본 데이터에서 해당 항목 찾기 (인덱스 기반 매칭으로 최적화)
            original_row = None
            # 첫 번째로 매칭되는 항목 사용 (이미 정렬된 상위 결과에서 선택했으므로)
            if i < len(rows):
                original_row = rows[i]
            
            if target_type == "brand":
                cid = f"brand:{name}"
            else:
                # 카테고리 타입일 때는 카테고리/상품 구분
                if not slots.focus and rec.get("type") == "category":
                    cid = f"category:{name}"
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
                "metrics": {k: original_row.get(k) for k in ("revenue","growth_pct","gm") if original_row and k in original_row} if original_row else {},
            }
            
            candidates.append(candidate)
            
            logger.info("  %d번 LLM 추천: %s (%s)", i+1, name, typ)
            logger.info("    - 근거 개수: %d", len(rec.get("reasons", [])))
            logger.info("    - 메트릭 요약: %s", rec.get("metrics_summary", "없음")[:100])

    option_json = {
        "candidates": candidates,
        "method": "simplified_v2",
        "time_window": "30days",
        "constraints": {},
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
    
    if not instructions:
        logger.warning("⚠️ instructions가 None입니다!")
        logger.warning("이전 노드에서 에러가 발생했을 가능성이 높습니다.")
        logger.warning("사용자 메시지: %s", state.get('user_message', 'N/A')[:100])
        return {"tool_results": {"error": "No instructions provided"}}
    
    tool_calls = instructions.tool_calls if instructions.tool_calls else []
    logger.info("instructions 정보: response_instruction='%s', tool_calls_count=%d", 
               instructions.response_generator_instruction[:100] if instructions.response_generator_instruction else 'N/A',
               len(tool_calls))

    if not tool_calls:
        logger.info("실행할 툴이 없습니다.")
        logger.info("instructions.tool_calls: %s", tool_calls)
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

            logger.info(f"🧩 {tool_name} 실행 - args: {tool_args}")
            
            if tool_name in tool_map:
                result_key = f"{tool_name}_{i}"
                future = executor.submit(tool_map[tool_name], tool_args)
                future_to_call[future] = result_key
                logger.info(f"✅ {tool_name} 제출 완료 (result_key: {result_key})")
            else:
                logger.warning(f"❌ 알 수 없는 도구 '{tool_name}' 호출은 건너뜁니다.")
                logger.warning(f"사용 가능한 도구: {list(tool_map.keys())}")
        
        for future in future_to_call:
            result_key = future_to_call[future]

            try:
                result = future.result()
                tool_results[result_key] = result
                logger.info(f"✅ {result_key} 실행 완료")
                # 결과 요약 로깅 (민감한 정보 제외)
                if isinstance(result, dict):
                    if 'rows' in result:
                        logger.info(f"  → {result_key} 데이터: {len(result.get('rows', []))}행")
                    elif 'results' in result:
                        logger.info(f"  → {result_key} 결과: {len(result.get('results', []))}건")
                    elif 'error' in result:
                        logger.warning(f"  → {result_key} 내부 에러: {result.get('error')}")

            except Exception as e:
                logger.error(f"❌ '{result_key}' 툴 실행 중 오류 발생: {e}", exc_info=True)
                tool_results[result_key] = {"error": str(e)}

    logger.info(f"툴 실행 완료: {len(tool_results)}개 결과")
    existing_results = state.get("tool_results") or {}
    merged_results = {**existing_results, **tool_results}
    
    logger.info(f"최종 tool_results 키: {list(merged_results.keys())}")
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
def safe_json_dumps(obj, **kwargs):
    def date_handler(obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')
    
    return json.dumps(obj, default=date_handler, **kwargs)

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
        json_data=safe_json_dumps(t2s_result, ensure_ascii=False)
    )
    
    viz_response = visualizer_app.invoke(viz_state)

    # 시각화 결과를 tool_results에 추가
    if viz_response:
        tool_results["visualization"] = {
            "json_graph": viz_response.get("json_graph"),
            "explanation": viz_response.get("output")
        }
        
    return {"tool_results": tool_results}

def promotion_final_generator(state: OrchestratorState, action_decision: dict, tr: dict) -> dict:
    """프로모션 최종 기획서 생성 (Claude-4-Sonnet 사용)"""
    logger.info("--- 🎯 프로모션 최종 기획서 생성 (Claude-4-Sonnet) ---")
    
    slots = state.get("active_task").slots if state.get("active_task") and state.get("active_task").slots else PromotionSlots()
    
    # 외부 데이터 수집 결과
    web_search = None
    marketing_trend_results = None
    youtuber_trend_results = None
    
    for key, value in tr.items():
        if key.startswith("tavily_search"): 
            web_search = value
        elif key.startswith("marketing_trend_search"):
            marketing_trend_results = value
        elif key.startswith("beauty_youtuber_trend_search"):
            youtuber_trend_results = value
    
    # 트렌드 반영 여부에 따른 프롬프트 구성
    has_trends = slots.wants_trend and (web_search or marketing_trend_results or youtuber_trend_results)
    
    prompt_tmpl = textwrap.dedent(f"""
    당신은 마케팅 전략 전문가입니다. 아래 정보를 바탕으로 **완성도 높은 프로모션 기획서**를 작성해 주세요.
    
    ## 📋 프로모션 기본 정보
    - 타겟 유형: {slots.target_type}
    - 포커스: {slots.focus} 
    - 타겟 고객층: {slots.target}
    - 선택 상품: {', '.join(slots.selected_product) if slots.selected_product else '없음'}
    - 기간: {slots.duration}
    - 목표: {slots.objective or '매출 증대'}
    
    {"## 🌟 수집된 트렌드 정보" if has_trends else ""}
    {f"### 웹 검색 결과: {web_search}" if web_search else ""}
    {f"### 마케팅 트렌드: {marketing_trend_results}" if marketing_trend_results else ""}
    {f"### 뷰티 트렌드: {youtuber_trend_results}" if youtuber_trend_results else ""}
    
    ## 🎯 작성 요구사항
    1. **프로모션 개요** (2-3줄로 핵심 컨셉 요약)
    2. **타겟 분석** (고객 특성과 니즈 분석)
    3. **핵심 메시지** (브랜드 메시지와 소구점)
    4. **실행 전략** (구체적인 마케팅 방법론)
    {"5. **트렌드 활용** (수집된 트렌드를 어떻게 활용할지)" if has_trends else ""}
    {"6." if has_trends else "5."} **예상 효과** (기대하는 성과와 KPI)
    {"7." if has_trends else "6."} **실행 일정** (주요 마일스톤)
    
    ## 📝 작성 가이드라인
    - 실무진이 바로 실행할 수 있는 구체적이고 실용적인 내용
    - 데이터와 근거 기반의 전략적 사고
    - 창의적이면서도 실현 가능한 아이디어
    {"- 최신 트렌드를 자연스럽게 녹여낸 현대적 접근" if has_trends else ""}
    - 한국어 존댓말로 전문적이고 세련된 톤앤매너
    
    완성도 높은 프로모션 기획서를 작성해 주세요.
    """)
    
    # Claude-4-Sonnet 사용
    # llm = ChatAnthropic(
    #     model="claude-sonnet-4-20250514", 
    #     temperature=0.1,
    #     max_tokens=8192,  # 넉넉한 토큰 제한 설정
    #     api_key=settings.ANTHROPIC_API_KEY
    # )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=settings.GOOGLE_API_KEY
    )
    
    final_text = llm.invoke(prompt_tmpl)
    final_response = getattr(final_text, "content", None) or str(final_text)
    
    logger.info("✅ Claude-4-Sonnet으로 프로모션 기획서 생성 완료")
    logger.info(f"프로모션 기획서:\n{final_response}")
    
    return final_response

def response_generator_node(state: OrchestratorState):
    logger.info("--- 🗣️ 응답 생성 노드 실행 ---")
    logger.info("입력: user_message='%s'", state.get('user_message', 'N/A')[:100])
    instructions = state.get("instructions")
    tr = state.get("tool_results") or {}
    logger.info("instructions 존재: %s, tool_results 키: %s", bool(instructions), list(tr.keys()))
    
    action_decision = tr.get("action")
    
    # 프로모션 최종 생성 상태들인지 확인
    if action_decision and action_decision.get("status") in ["create_final_plan", "apply_trends"]:
        final_response = promotion_final_generator(state, action_decision, tr)
        
        history = state.get("history", [])
        history.append({"role": "user", "content": state.get("user_message", "")})
        history.append({"role": "assistant", "content": final_response})
        
        # 프로모션 슬롯 정보도 함께 반환 (plan 데이터 생성용)
        slots = state.get("active_task").slots if state.get("active_task") and state.get("active_task").slots else PromotionSlots()
        
        return {
            "history": history, 
            "user_message": "", 
            "output": final_response,
            "promotion_slots": slots.model_dump(),
            "is_final_promotion": True
        }

    # 기존 로직 (Gemini 사용)
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
    1) **가장 중요한 규칙**: `action_decision` 객체가 있고, 그 안의 `ask_prompts` 리스트에 내용이 있다면, 당신의 최우선 임무는 해당 리스트의 질문을 사용자에게 하는 것입니다. 다른 모든 지시보다 이 규칙을 **반드시** 따라야 합니다. `ask_prompts`의 문구를 그대로 사용하거나, 살짝 더 자연스럽게만 다듬어 질문하세요.
    1-1) **중복 질문 방지**: 이미 채워진 슬롯에 대해서는 절대 재질문하지 마세요. ask_prompts에 있어도 이미 답변된 내용이면 건너뛰세요.
    1-2) **진행 상황 정확히 파악**: action_decision의 payload나 missing_slots를 보고 현재 몇 번째 질문인지 정확히 판단하세요. 첫 질문이면 "현재까지 수집했다"는 식의 표현을 사용하지 마세요.
    1-3) **"마지막" 표현 주의**: 프로모션 질문 순서는 ①프로모션종류 ②브랜드/카테고리 ③기간 ④상품선택 ⑤트렌드반영 입니다. 트렌드반영 질문을 할 때만 "마지막으로"라는 표현을 사용하세요.
    2) **프로모션 완성 규칙**: 
       - `action_decision`의 `status`가 "start_promotion"인 경우, 완성된 프로모션 슬롯 정보를 기반으로 프로모션 내용을 정리해서 보여주고, 마지막 문단에 반드시 "최신 트렌드나 유행어를 반영해서 프로모션을 만들길 원하시나요?"라고 질문하세요.
       - `action_decision`의 `status`가 "create_final_plan"인 경우, 트렌드 반영 없이 완성된 프로모션 기획서를 제작하세요.
       - `action_decision`의 `status`가 "apply_trends"이고 외부 데이터가 있는 경우, 수집된 트렌드를 반영한 최종 프로모션 기획서를 제작하세요.
       - **중요**: 사용자가 이미 트렌드 반영 여부에 대해 답변했다면 (wants_trend가 true/false로 설정됨), 같은 질문을 다시 하지 마세요.
    3) 위 1,2번 규칙에 해당하지 않는 경우에만, `instructions_text`를 주된 내용으로 삼아 답변을 생성합니다.
    4) **프로모션 필드 질문 규칙**: 
       - 필수 필드: target_type, focus, duration, selected_product (반드시 물어봐야 함)
       - wants_trend: 트렌드 반영 질문만 (사용자가 먼저 언급하지 않는 한 굳이 물어보지 마세요)
       - objective: 사용자가 명시적으로 언급한 경우에만 처리 (굳이 질문하지 마세요)
       - 금지 필드: budget, cost, 예산 등 (존재하지 않는 필드들)
       - **중요**: missing_slots 리스트를 확인해서 남은 필드가 얼마나 있는지 파악하고, 적절한 톤으로 질문하세요.
    5) `option_candidates`가 있으면 번호로 제시하고 각 2~4줄 근거를 붙입니다. 
       - 후보에 `llm_reasons` 필드가 있으면 그것을 우선 사용하세요 (LLM이 생성한 상세 근거)
       - `llm_reasons`가 없으면 기존 `reasons`, `business_reasons` 등을 사용하세요
       - 모든 수치는 어떤 수치인지 구체적인 언급을 해주세요
       - 마지막에 '기타(직접 입력)'도 추가합니다    
    6) web_search / scraped_pages / supabase 결과가 있으면, 핵심 근거를 2~4줄로 요약해 설명에 녹여 주세요. 원문 인용은 1~2문장 이하로 제한.
    7) t2s_table 처리 규칙:
       - output_type이 "export"인 경우: 표나 시각화를 포함하지 말고, 데이터 준비가 완료되었음을 안내하세요. 다운로드 링크는 시스템에서 자동으로 추가됩니다.
       - output_type이 "table"인 경우: 상위 10행 미리보기 표만 포함하되, 없는 수치는 만들지 마세요. 표를 시작하는 부분은 [TABLE_START] 표가 끝나는 부분은 [TABLE_END] 라는 텍스트를 붙여서 어디부터 어디가 테이블인지 알 수 있게 해주세요.
       - output_type이 "visualize"인 경우: 상위 10행 미리보기 표를 포함하고, 시각화 결과가 있다면 함께 제공하세요.
    8) 전체적으로 구조화된 형식을 유지하세요.

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

    def to_json(x):
        if x is None:
            return "null"
        try:
            return json.dumps(x, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"JSON 직렬화 실패: {e}, 빈 객체로 처리")
            return "{}"

    prompt = ChatPromptTemplate.from_template(prompt_tmpl)
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=settings.GOOGLE_API_KEY)
    # llm = ChatAnthropic(
    #     model="claude-sonnet-4-20250514", 
    #     temperature=0.1,
    #     max_tokens=8192,  # 넉넉한 토큰 제한 설정
    #     api_key=settings.ANTHROPIC_API_KEY
    # )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=settings.GOOGLE_API_KEY
    )
        
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

    # AIMessage 객체 안전 처리
    if hasattr(final_text, "content"):
        final_response = final_text.content
    else:
        final_response = str(final_text)
    
    logger.info(f"✅ 응답 생성 완료 (길이: {len(final_response)}자)")
    logger.info(f"최종 결과 미리보기: {final_response}...")
    
    history = state.get("history", [])
    history.append({"role": "user", "content": state.get("user_message", "")})
    history.append({"role": "assistant", "content": final_response})
    
    logger.info(f"히스토리 업데이트: 총 {len(history)}개 메시지")
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
        "tool_executor": "tool_executor",
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