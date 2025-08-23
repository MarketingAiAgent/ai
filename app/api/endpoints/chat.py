from fastapi import APIRouter, Body, Path, HTTPException
from fastapi.responses import StreamingResponse
from starlette import status
import json
import asyncio
import uuid 

from app.agents.__init__ import stream_agent
from app.agents.orchestrator.state import PromotionSlots, ActiveTask
from app.schema.chat import ChatRequest, NewChatRequest, CreatePlanRequest
from app.core.config import settings
from app.database.chat_history import *
from app.database.promotion_slots import get_or_create_state
from app.service.chat_service import generate_chat_title, stream_and_save_wrapper
from app.mock import get_mock_response, mock_stream_with_save

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/stream/new")
async def new_chat_stream(request: NewChatRequest = Body(...)):

    # Mock 응답 체크 (한 줄로 처리)
    if mock_response := get_mock_response(request.user_message):
        chat_title = await generate_chat_title(request.user_message)
        chat_id = crete_chat(user_id=request.user_id, title=chat_title)
        
        # Mock 응답을 스트리밍하면서 채팅 히스토리에 저장
        final_stream = mock_stream_with_save(chat_id, request.user_message, mock_response)
        return StreamingResponse(final_stream, media_type="text/event-stream")

    chat_title = await generate_chat_title(request.user_message)
    chat_id = crete_chat(user_id=request.user_id, title=chat_title)
    
    history = []  
    response_stream = stream_agent(
        chat_id=chat_id,
        history=history, 
        active_task=None, 
        conn_str=settings.CONN_STR, 
        schema_info=settings.SCHEMA_INFO, 
        message=request.user_message
    )

    async def initial_event_stream(agent_stream):
        initial_payload = {
            "type": "chat_id",
            "content": chat_id,
        }
        yield f"data: {json.dumps(initial_payload, ensure_ascii=False)}\n\n"
        
        async for chunk in agent_stream:
            yield chunk
    
    final_stream_with_info = initial_event_stream(response_stream)
    final_stream_to_save = stream_and_save_wrapper(chat_id, request.user_message, final_stream_with_info)

    return StreamingResponse(final_stream_to_save, media_type="text/event-stream")

@router.post("/stream")
async def chat_stream(request: ChatRequest = Body(...)):

    # Mock 응답 체크 (한 줄로 처리)
    if mock_response := get_mock_response(request.user_message):
        # Mock 응답을 스트리밍하면서 채팅 히스토리에 저장
        final_stream = mock_stream_with_save(request.chat_id, request.user_message, mock_response)
        return StreamingResponse(final_stream, media_type="text/event-stream")

    history = get_chat_history(chat_id=request.chat_id)
    slots = get_or_create_state(chat_id=request.chat_id)

    current_active_task = ActiveTask(
        task_id=request.chat_id,
        status="in_progress",
        slots=PromotionSlots(**slots)
    )

    response_stream = stream_agent(
        chat_id=request.chat_id,
        history=history, 
        active_task=current_active_task, 
        conn_str=settings.CONN_STR, 
        schema_info=settings.SCHEMA_INFO, 
        message=request.user_message
    )
    
    final_stream = stream_and_save_wrapper(request.chat_id, request.user_message, response_stream)

    return StreamingResponse(final_stream, media_type="text/event-stream")

@router.delete("/{chat_id}", summary="Delete Chat History")
def delete_chat(chat_id: str = Path(...)):
    deleted_count = delete_chat_history(chat_id=chat_id)

    if deleted_count:
        return {
            "message": "Chat history deleted successfully.",
            "chat_id": chat_id,
        }

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat history with thread_id '{chat_id}' not found."
        )

@router.post("/createPlan")
def create_plan(request: CreatePlanRequest):
    return mock_create_plan()