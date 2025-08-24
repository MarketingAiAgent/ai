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
from app.mock.plan import mock_create_plan
from app.service.stream import stream_agent_v2

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/new")
async def new_chat_stream(request: NewChatRequest = Body(...)):

    chat_title = await generate_chat_title(request.message)
    chat_id = crete_chat(user_id=request.user_id, title=chat_title)

    return {"chatId": chat_id}

@router.post("/stream")
async def chat_stream(request: ChatRequest = Body(...)):

    if mock_response := get_mock_response(request.user_message):
        final_stream = mock_stream_with_save(request.chat_id, request.user_message, mock_response)
        return StreamingResponse(final_stream, media_type="text/event-stream")

    history = get_chat_history(chat_id=request.chat_id)
    slots = get_or_create_state(chat_id=request.chat_id)

    current_active_task = ActiveTask(
        task_id=request.chat_id,
        status="in_progress",
        slots=PromotionSlots(**slots)
    )

    sql_context = {
        "conn_str": settings.CONN_STR,
        "schema_info": settings.SCHEMA_INFO
    }
    
    response_stream = stream_agent_v2(
        chat_id=request.chat_id,
        history=history, 
        user_message=request.user_message,
        sql_context=sql_context,
        promotion_slots=slots
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
    chat_id = request.chat_id 
    active_state = get_or_create_state(chat_id=chat_id)

    return_type = active_state['target_type']
    if return_type != "brand" and return_type != "category":
        return_type = 'brand'

    return mock_create_plan(return_type, request.company)

    # if active_tast.status == "in_progress":
    #     raise HTTPException(
    #         status_code=status.HTTP_404_NOT_FOUND,
    #         detail=f"아직 프로모션이 준비되지 않았습니다."
    #     )
    # else: