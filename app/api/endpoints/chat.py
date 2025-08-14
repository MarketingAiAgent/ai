from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
import json
import asyncio

from app.agents.__init__ import stream_agent
from app.schema.chat import ChatRequest
from app.core.config import settings

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/stream")
async def chat_stream(request_data: ChatRequest = Body(...)):

    response_stream = stream_agent(history=[], active_task=None, conn_str=settings.CONN_STR, schema_info=settings.SCHEMA_INFO, message=request_data.user_message)
    return StreamingResponse(response_stream, media_type="text/event-stream")

# async def mock_agent_stream(request_data: ChatRequest):
#     """
#     [최종] 리포트의 텍스트 부분을 한 글자씩 스트리밍하고,
#     그래프를 같은 말풍선 안에 표시하는 최종 버전
#     """
#     # 1. 리포트 시작을 알림
#     yield f"data: {json.dumps({'type': 'report_start'})}\n\n"
#     await asyncio.sleep(0.5)

#     # 2. 리포트의 Markdown 텍스트 부분을 한 글자씩 스트리밍
#     report_markdown = """
# ### 월별 매출 분석 리포트

# **분석 요약:**
# - 2월에 매출이 15로 가장 높게 나타났습니다.
# - 3월에는 매출이 소폭 하락했습니다.

# **향후 제안:**
# 1. 2월의 성공 요인을 분석하여 4월 마케팅에 적용
# 2. 3월 매출 하락 원인 파악을 위한 추가 데이터 분석 필요
# """
#     for char in report_markdown:
#         text_chunk = {"type": "text_chunk", "content": char}
#         yield f"data: {json.dumps(text_chunk)}\n\n"
#         await asyncio.sleep(0.02) # 타이핑 효과를 위한 매우 짧은 딜레이

#     # 3. 그래프 데이터를 전송
#     plotly_figure = {
#         "data": [{"x": ["1월", "2월", "3월"], "y": [10, 15, 13], "type": "bar"}],
#         "layout": {"title": "월별 매출 현황"}
#     }
#     graph_message = {"type": "graph", "content": plotly_figure}
#     yield f"data: {json.dumps(graph_message)}\n\n"

#     # 4. 종료 신호 전송
#     done_message = {"type": "done"}
#     yield f"data: {json.dumps(done_message)}\n\n"
    