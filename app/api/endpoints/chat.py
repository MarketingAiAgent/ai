from fastapi import APIRouter, Body, Path, HTTPException
from fastapi.responses import StreamingResponse
from starlette import status
import json
import asyncio
import uuid 

from app.agents.__init__ import stream_agent
from app.agents.orchestrator.state import PromotionSlots, ActiveTask
from app.agents.formatter.grapy import create_plan_from_promotion_slots
from app.schema.chat import ChatRequest, NewChatRequest, CreatePlanRequest
from app.core.config import settings
from app.database.chat_history import *
from app.database.promotion_slots import get_or_create_state
from app.service.chat_service import generate_chat_title, stream_and_save_wrapper
from app.mock import get_mock_response, mock_stream_with_save 
from app.mock.plan import mock_create_plan

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/new")
async def new_chat_stream(request: NewChatRequest = Body(...)):

    chat_title = await generate_chat_title(request.message)
    chat_id = crete_chat(user_id=request.user_id, title=chat_title)

    return {"chatId": chat_id}

@router.post("/stream")
async def chat_stream(request: ChatRequest = Body(...)):

    if mock_response := get_mock_response(request.user_message, request.chat_id):
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
    chat_id = request.chat_id 
    active_state = get_or_create_state(chat_id=chat_id)

    # 프로모션 슬롯 데이터 추출
    promotion_slots = {
        'target_type': active_state.get('target_type', 'brand'),
        'focus': active_state.get('focus'),
        'target': active_state.get('target'),
        'objective': active_state.get('objective'),
        'duration': active_state.get('duration'),
        'selected_product': active_state.get('selected_product', []),
        'product_options': active_state.get('product_options', []),
        'wants_trend': active_state.get('wants_trend')
    }
    
    # 최근 채팅 히스토리에서 프로모션 기획 내용 추출
    history = get_chat_history(chat_id=chat_id)
    promotion_content = ""
    
    # 최근 어시스턴트 메시지에서 프로모션 내용 찾기
    for message in reversed(history):
        if message.get('role') == 'assistant':
            content = message.get('content', '')
            if content and len(content) > 100:  # 충분히 긴 메시지가 프로모션 기획서일 가능성
                promotion_content = content
                break
    
    # 프로모션 내용이 없으면 기본 텍스트 사용
    if not promotion_content:
        target_type = promotion_slots.get('target_type', 'brand')
        focus = promotion_slots.get('focus', '브랜드')
        target = promotion_slots.get('target', '고객')
        promotion_content = f"{focus} {target} 프로모션 기획서"
    
    try:
        # formatter를 통해 실제 plan 데이터 생성
        plan_data = create_plan_from_promotion_slots(promotion_slots, promotion_content)
        
        # plan_data에서 final_exam 추출하여 반환
        final_exam = plan_data.get('final_exam', [])
        if final_exam:
            response = final_exam[0]
            response['type'] = promotion_slots.get('target_type', 'brand')
            return final_exam[0]  # 첫 번째 기획서 반환
        else:
            # fallback: mock 데이터 사용
            return_type = promotion_slots.get('target_type', 'brand')
            return mock_create_plan(return_type, request.company, request.user_id)
            
    except Exception:
        # 에러 발생 시 mock 데이터 사용
        return_type = promotion_slots.get('target_type', 'brand')
        return mock_create_plan(return_type, request.company, request.user_id)