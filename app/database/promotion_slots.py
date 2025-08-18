from datetime import datetime
from .connection import db  
from pymongo.results import UpdateResult
import logging

from app.agents.orchestrator.state import PromotionSlots

logger = logging.getLogger(__name__)

def get_or_create_state(thread_id: str) -> dict:
    if db is None:
        raise ConnectionError("DB에 연결되지 않았습니다.")

    try:
        collection = db.conversation_states
        state = collection.find_one({"thread_id": thread_id})

        if state:
            logger.info(f"✅ 기존 상태를 불러왔습니다. (대화방 ID: {thread_id})")
            return state
        else:
            logger.info(f"✨ 새로운 상태를 생성합니다. (대화방 ID: {thread_id})")

            default_state = PromotionSlots()
            new_state = default_state.model_dump()
            new_state['thread_id'] = thread_id
            
            collection.insert_one(new_state)
            return new_state
            
    except Exception as e:
        logger.error(f"❌ 상태 조회/생성 중 오류 발생: {e}")
        return { "thread_id": thread_id }


def update_state(thread_id: str, new_values: dict) -> UpdateResult:
    if db is None:
        raise ConnectionError("DB에 연결되지 않았습니다.")

    try:
        collection = db.conversation_states
        
        result = collection.update_one(
            {"thread_id": thread_id},
            {"$set": new_values}
        )
        
        if result.modified_count > 0:
            logger.info(f"🔄 상태가 업데이트되었습니다. (채팅방 ID: {thread_id}, 변경: {new_values})")
        
        return result

    except Exception as e:
        logger.error(f"❌ 상태 업데이트 중 오류 발생: {e}")
        return None