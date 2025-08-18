from fastapi import APIRouter, Body, Path, HTTPException
from fastapi.responses import StreamingResponse
from starlette import status
import json
import asyncio

from app.agents.__init__ import stream_agent
from app.schema.chat import ChatRequest
from app.core.config import settings
from app.database.chat_history import get_chat_history, save_chat_message, delete_chat_history

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

@router.delete("/{thread_id}", summary="Delete Chat History")
async def delete_chat(thread_id: str = Path(...)):
    deleted_count = delete_chat_history(session_id=thread_id)

    if deleted_count > 0:
        return {
            "message": "Chat history deleted successfully.",
            "thread_id": thread_id,
            "deleted_messages": deleted_count
        }

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat history with thread_id '{thread_id}' not found."
        )