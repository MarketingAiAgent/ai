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

    try:
        async for event in orchestrator_app.astream_events(state, version="v2"):
            kind = event["event"]
            current_node = event.get("metadata", {}).get("langgraph_node")
            
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
        if graph:
            yield f"data: {json.dumps({'type':graph, 'content':graph})}"
            
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"