import json
import time

from langchain_google_genai import ChatGoogleGenerativeAI

from .orchestrator.state import return_initial_state
from .orchestrator.graph import orchestrator_app

def stream_agent(history, active_task, conn_str, schema_info, message): 

    state = return_initial_state(history, active_task, conn_str, schema_info,message)

    yield f"data: {json.dumps({'type' : 'report_start'})}\n\n"

    for chunk, meta_data in orchestrator_app.stream(state, stream_mode='messages'):
        if chunk.content: 
            for char in chunk.content:
                char_chunk_data = {"type": "text_chunk", 'content': char}
                yield f"data: {json.dumps(char_chunk_data)}\n\n"

                time.sleep(0.02)

    yield f"data: {json.dumps({'type': 'done'})}\n\n"

# def stream_example_agent(message: str):
#     """
#     LLM의 응답을 '한 글자 단위'로 쪼개 SSE 형식으로 스트리밍합니다.
#     """
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash", 
#         temperature=0, 
#         google_api_key=settings.GOOGLE_API_KEY
#     )
    
#     yield f"data: {json.dumps({'type': 'report_start'})}\n\n"

#     for chunk in llm.stream(message):
#         if chunk.content:
#             # 💡 핵심: 받은 데이터 덩어리(chunk)를 한 글자씩(char) 순회합니다.
#             for char in chunk.content:
#                 # 한 글자를 JSON 형식으로 감싸서 전송합니다.
#                 char_chunk_data = {'type': 'text_chunk', 'content': char}
#                 yield f"data: {json.dumps(char_chunk_data)}\n\n"
                
#                 # (선택 사항) 타이핑 효과를 더 명확하게 보려면 아주 작은 딜레이를 추가할 수 있습니다.
#                 time.sleep(0.02)

#     yield f"data: {json.dumps({'type': 'done'})}\n\n"