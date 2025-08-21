import json
import logging
from collections import deque 

from .orchestrator.state import return_initial_state
from .orchestrator.graph import orchestrator_app

async def stream_agent(thread_id, history, active_task, conn_str, schema_info, message):
    state = return_initial_state(thread_id, history, active_task, conn_str, schema_info, message)

    yield f"data: {json.dumps({'type': 'start'}, ensure_ascii=False)}\n\n"

    graph = None
    is_in_table = False 
    
    TOKEN_START = "[TABLE_START]"
    TOKEN_END = "[TABLE_END]"

    buffer_outside_table = deque(maxlen=len(TOKEN_START))
    buffer_inside_table = deque(maxlen=len(TOKEN_END))

    buffer = None 
    
    TOOL_NAME_MAP = {
        "t2s": "데이터베이스 조회",
        "tavily_search": "웹 검색",
        "scrape_webpages": "웹페이지 분석",
        "marketing_trend_search": "마케팅 트렌드 분석",
        "beauty_youtuber_trend_search": "뷰티 트렌드 분석",
    }

    try:
        async for event in orchestrator_app.astream_events(state, version="v2"):
            kind = event["event"]
            current_node = event.get("metadata", {}).get("langgraph_node")
            
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
                        "content": f"{display_name} 실행 중..."
                    }
                    yield f"data: {json.dumps(tool_payload, ensure_ascii=False)}\n\n"
                
                # 구체적인 툴 메시지를 보냈으므로, 제네릭한 노드 메시지는 건너뜁니다.
                continue
                
            if kind=='on_chain_start':
                if current_node not in ["__start__", "__end__"]:
                    state_payload = {
                        "type": "state",
                        "content": f"{current_node} 노드 수행 중..."
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