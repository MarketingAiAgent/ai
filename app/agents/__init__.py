import json
import time

from langchain_google_genai import ChatGoogleGenerativeAI

from .orchestrator.state import return_initial_state
from .orchestrator.graph import orchestrator_app

# def stream_agent(history, active_task, conn_str, schema_info, message): 

#     state = return_initial_state(history, active_task, conn_str, schema_info,message)

#     yield f"data: {json.dumps({'type' : 'report_start'})}\n\n"

#     for chunk, metadata in orchestrator_app.stream(state, stream_mode='messages'):
#         if chunk.content: 
#             for char in chunk.content:
#                 char_chunk_data = {"type": "text_chunk", 'content': char}
#                 yield f"data: {json.dumps(char_chunk_data)}\n\n"

#                 time.sleep(0.02)

#     yield f"data: {json.dumps({'type': 'done'})}\n\n"

import json, time

def _json_safe(obj):
    def _fallback(o):
        md = getattr(o, "model_dump", None)
        if callable(md):
            return md()
        mj = getattr(o, "model_dump_json", None)
        if callable(mj):
            try:
                return json.loads(mj())
            except Exception:
                return str(o)
        return str(o)
    try:
        return json.dumps(obj, ensure_ascii=False, default=_fallback)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)

def _normalize_stream_payload(payload):
    if isinstance(payload, tuple):
        update = payload[0] if len(payload) >= 1 else {}
        meta = payload[1] if len(payload) >= 2 and isinstance(payload[1], dict) else {}
    else:
        update, meta = payload, {}
    return update, meta

async def stream_agent(history, active_task, conn_str, schema_info, message):
    state = return_initial_state(history, active_task, conn_str, schema_info, message)

    yield f"data: {json.dumps({'type': 'report_start'}, ensure_ascii=False)}\n\n"

    acc_state = {}

    try:
        for payload in orchestrator_app.stream(state, stream_mode='values'):
            update, _meta = _normalize_stream_payload(payload)
            if isinstance(update, dict):
                acc_state.update(update)

        final_text = ""
        if "output" in acc_state and acc_state["output"] is not None:
            val = acc_state["output"]
            final_text = val if isinstance(val, str) else _json_safe(val)
        else:
            hist = acc_state.get("history") or []
            if hist and isinstance(hist[-1], dict) and hist[-1].get("role") == "assistant":
                final_text = str(hist[-1].get("content") or "")

        if final_text:
            for ch in final_text:
                yield f"data: {json.dumps({'type': 'text_chunk', 'content': ch}, ensure_ascii=False)}\n\n"
                time.sleep(0.02)

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"