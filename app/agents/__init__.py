import json
import logging
from collections import deque 

from .orchestrator.state import return_initial_state
from .orchestrator.graph import orchestrator_app
from .formatter.grapy import create_plan_from_promotion_slots
from app.mock.chat import *

async def stream_agent(chat_id, history, active_task, conn_str, schema_info, message):
    state = return_initial_state(chat_id, history, active_task, conn_str, schema_info, message)

    yield f"data: {json.dumps({'type': 'start'}, ensure_ascii=False)}\n\n"

    graph = None
    is_in_table = False 
    
    TOKEN_START = "[TABLE_START]"
    TOKEN_END = "[TABLE_END]"

    buffer_outside_table = deque(maxlen=len(TOKEN_START))
    buffer_inside_table = deque(maxlen=len(TOKEN_END))

    buffer = None 
    
    TOOL_NAME_MAP = {
        "t2s": "데이터베이스 조회 중...",
        "tavily_search": "웹 검색 중...",
        "scrape_webpages": "웹페이지 내용 추출 중...",
        "marketing_trend_search": "마케팅 트렌드 지식DB 조회 중...",
        "beauty_youtuber_trend_search": "뷰티 트렌드 지식DB 조회 중...",
    }
    
    NODE_NAME_MAP = {
        "generate_sql": "SQL 생성 중...",
        "make_table": "테이블 생성 중...",
        "planner": "응답 계획 수립 중...",
        "slot_extractor": "프로모션 구성 시작...",
        "action_state": "다음 행동 선택 중...",
        "options_generator": "선택지 탐색 중...",
        "tool_executor": "정보 탐색 툴 선택 중...",
        "visualizer": "그래프 생성 중...",
        "response_generator": "응답 생성 중...",
    }

    try:
        async for event in orchestrator_app.astream_events(state, version="v2"):
            kind = event["event"]
            current_node = event.get("metadata", {}).get("langgraph_node")
            
            # 프로모션 최종 생성 완료 시 plan 데이터 전송
            if kind == "on_chain_end" and current_node == "response_generator":
                final_state = event.get("data", {}).get("output")
                if final_state and final_state.get("is_final_promotion"):
                    # 프로모션 슬롯과 기획 내용 추출
                    promotion_slots = final_state.get("promotion_slots", {})
                    promotion_content = final_state.get("output", "")
                    
                    # formatter를 통해 plan 데이터 생성
                    try:
                        create_plan_from_promotion_slots(promotion_slots, promotion_content)
                        plan_payload = {
                            "type": "plan",
                            "content": promotion_slots.get('target_type', 'brand')
                        }
                        yield f"data: {json.dumps(plan_payload, ensure_ascii=False)}\n\n"
                    except Exception as e:
                        logging.error(f"Plan data generation failed: {e}")
                        # 실패 시에도 기본 plan 데이터 전송
                        plan_payload = {
                            "type": "plan",
                            "content": promotion_slots.get('target_type', 'brand')
                        }
                        yield f"data: {json.dumps(plan_payload, ensure_ascii=False)}\n\n"
            
            if kind == "on_chain_end" and current_node== "visualizer":
                final_state = event.get("data", {}).get("output")
                if final_state:
                    viz_data = final_state.get("tool_results", {}).get("visualization")
                    if viz_data and viz_data.get("json_graph"):
                        graph = {
                            "type": "graph",
                            "content": viz_data["json_graph"]
                        }

            if current_node == "tool_executor":
                # 노드로 들어가는 입력(state)에서 tool_calls를 추출합니다.
                input_state = event.get("data", {}).get("input", {})
                instructions = input_state.get("instructions")
                tool_calls = instructions.tool_calls if instructions else []
                
                if not tool_calls:
                    continue # 실행할 툴이 없으면 넘어감

                # 각 툴 호출에 대해 구체적인 메시지를 전송합니다.
                for call in tool_calls:
                    tool_name = call.get("tool", "알 수 없는 툴")
                    display_name = TOOL_NAME_MAP.get(tool_name, tool_name)
                    tool_payload = {
                        "type": "state",
                        "content": f"{display_name}"
                    }
                    yield f"data: {json.dumps(tool_payload, ensure_ascii=False)}\n\n"
                
                # 구체적인 툴 메시지를 보냈으므로, 제네릭한 노드 메시지는 건너뜁니다.
                continue
                
            if kind=='on_chain_start':
                if current_node not in ["__start__", "__end__"]:
                    node_display_name = NODE_NAME_MAP.get(current_node, current_node)
                    state_payload = {
                        "type": "state",
                        "content": f"{node_display_name}"
                    }
                    yield f"data: {json.dumps(state_payload, ensure_ascii=False)}\n\n"
            
            if kind == "on_chat_model_stream" and current_node== "response_generator":
                chunk = event.get("data", {}).get("chunk")
                if chunk and chunk.content:
                    for c in chunk.content:

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

    except Exception as e:
        error_payload = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    finally:
        if buffer:
            for c in buffer: 
                yield f"data: {json.dumps({'type': 'chunk', 'content': c}, ensure_ascii=False)}\n\n"
        if graph:
            yield f"data: {json.dumps(graph, ensure_ascii=False)}\n\n"
            
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"