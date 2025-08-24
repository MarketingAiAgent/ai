import json
import logging
from collections import deque
from typing import AsyncGenerator, Dict, Any

from app.agents_v2.orchestrator.state import AgentState, PromotionSlots
from app.agents_v2.orchestrator.graph import orchestrator_app

logger = logging.getLogger(__name__)

async def stream_agent_v2(
    chat_id: str,
    history: list,
    user_message: str,
    sql_context: Dict[str, Any],
    promotion_slots: Dict[str, Any] = None
) -> AsyncGenerator[str, None]:
    """
    agents_v2를 위한 스트리밍 에이전트
    
    Args:
        chat_id: 채팅 ID
        history: 대화 히스토리
        user_message: 사용자 메시지
        sql_context: SQL 실행 컨텍스트 (conn_str, schema_info)
        promotion_slots: 기존 프로모션 슬롯 (선택)
    
    Yields:
        Server-Sent Events 형식의 JSON 문자열
    """
    
    # promotion_slots를 PromotionSlots 객체로 변환
    promotion_slots_obj = None
    if promotion_slots:
        try:
            promotion_slots_obj = PromotionSlots.model_validate(promotion_slots)
        except Exception as e:
            logger.warning(f"프로모션 슬롯 변환 실패: {e}")
    
    # 초기 상태 생성
    initial_state = AgentState(
        history=history,
        user_message=user_message,
        promotion_slots=promotion_slots_obj,
        sql_context=sql_context
    )
    
    yield f"data: {json.dumps({'type': 'start'}, ensure_ascii=False)}\n\n"
    
    # 상태 추적
    graph = None
    is_in_table = False
    
    # 테이블 토큰 처리
    TOKEN_START = "[TABLE_START]"
    TOKEN_END = "[TABLE_END]"
    
    buffer_outside_table = deque(maxlen=len(TOKEN_START))
    buffer_inside_table = deque(maxlen=len(TOKEN_END))
    buffer = None
    
    # 노드별 표시 이름 매핑
    NODE_NAME_MAP = {
        "router": "의도 분석",
        "slot_extractor": "정보 추출",
        "ask_scope_period": "기본 정보 수집",
        "plan_options": "옵션 계획 수립",
        "ask_target": "타겟 추천",
        "promotion_report": "프로모션 리포트 생성",
        "qa_plan": "QA 계획 수립",
        "qa_run": "데이터 분석 실행",
        "qa_answer": "답변 생성",
        "out_of_scope": "범위 외 처리"
    }
    
    try:
        async for event in orchestrator_app.astream_events(initial_state, version="v2"):
            kind = event["event"]
            current_node = event.get("metadata", {}).get("langgraph_node")
            
            # 노드 시작 시 상태 메시지 전송
            if kind == "on_chain_start":
                if current_node not in ["__start__", "__end__"]:
                    display_name = NODE_NAME_MAP.get(current_node, current_node)
                    state_payload = {
                        "type": "state",
                        "content": f"{display_name} 중..."
                    }
                    yield f"data: {json.dumps(state_payload, ensure_ascii=False)}\n\n"
            
            # 노드 완료 시 결과 처리
            if kind == "on_chain_end":
                final_state = event.get("data", {}).get("output")
                if not final_state:
                    continue
                
                # QA 관련 처리
                if current_node == "qa_run":
                    # 시각화 데이터 처리
                    if final_state.qa_chart:
                        graph = {
                            "type": "graph",
                            "content": final_state.qa_chart
                        }
                
                # 최종 응답 처리 (최종 노드들만)
                if current_node in ["qa_answer", "out_of_scope", "ask_scope_period", "ask_target", "promotion_report"] and final_state.response:
                    # 테이블 토큰 처리
                    for c in final_state.response:
                        if is_in_table:
                            buffer = buffer_inside_table
                            token = TOKEN_END
                            alert = "table_end"
                        else:
                            buffer = buffer_outside_table
                            token = TOKEN_START
                            alert = "table_start"
                        
                        if len(buffer) < buffer.maxlen:
                            buffer.append(c)
                            continue
                        
                        yield f"data: {json.dumps({'type': 'chunk', 'content': buffer.popleft()}, ensure_ascii=False)}\n\n"
                        buffer.append(c)
                        
                        window_text = "".join(buffer)
                        if window_text == token:
                            is_in_table = not is_in_table
                            yield f"data: {json.dumps({'type': alert}, ensure_ascii=False)}\n\n"
                            buffer.clear()
            
            # 에러 처리
            if kind == "on_chain_error":
                error_data = event.get("data", {})
                error_msg = error_data.get("error", "알 수 없는 오류가 발생했습니다.")
                error_payload = {
                    "type": "error",
                    "message": error_msg
                }
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
    
    except Exception as e:
        logger.exception("stream_agent_v2 실행 중 오류 발생")
        error_payload = {
            "type": "error",
            "message": f"시스템 오류가 발생했습니다: {str(e)}"
        }
        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
    
    finally:
        # 버퍼에 남은 문자들 전송
        if buffer:
            for c in buffer:
                yield f"data: {json.dumps({'type': 'chunk', 'content': c}, ensure_ascii=False)}\n\n"
        
        # 그래프 데이터 전송
        if graph:
            yield f"data: {json.dumps(graph, ensure_ascii=False)}\n\n"
        
        # 완료 신호 전송
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

