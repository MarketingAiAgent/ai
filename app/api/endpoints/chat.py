from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
import json
import asyncio

from app.agents.__init__ import stream_agent
from app.schema.chat import ChatRequest
from app.core.config import settings
from app.database.chat_history import get_chat_history, save_chat_message

router = APIRouter(prefix="/chat", tags=["Chat"])

async def stream_and_save(request, response_stream):
    full_response = [] 

    async for chunk in response_stream:
        yield chunk 
        full_response.append(chunk)
    
    final_response = "".join(full_response)

    save_chat_message(
        thread_id=request.thread_id, 
        user_message=request.user_message,
        agent_message=final_response
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest = Body(...)):

    history = get_chat_history(thread_id=request.thread_id)

    response_stream = stream_agent(
        history=history, 
        active_task=None, 
        conn_str=settings.CONN_STR, 
        schema_info=settings.SCHEMA_INFO, 
        message=request.user_message
        )
    
    final_stream = stream_and_save(request, response_stream)

    return StreamingResponse(final_stream, media_type="text/event-stream")