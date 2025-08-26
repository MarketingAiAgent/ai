from datetime import datetime
from .connection import db  
from pymongo.results import UpdateResult
import logging

from app.agents.orchestrator.state import PromotionSlots

logger = logging.getLogger(__name__)

def get_or_create_state(chat_id: str) -> dict:
    if db is None:
        raise ConnectionError("DB에 연결되지 않았습니다.")

    try:
        collection = db.states
        state = collection.find_one({"chat_id": chat_id})

        if state:
            logger.info(f"✅ 기존 상태를 불러왔습니다. (대화방 ID: {chat_id})")
            return state
        else:
            logger.info(f"✨ 새로운 상태를 생성합니다. (대화방 ID: {chat_id})")

            default_state = PromotionSlots()
            new_state = default_state.model_dump(exclude_none=False)  # None 값도 포함하도록 수정
            new_state['chat_id'] = chat_id
            new_state['created_at'] = datetime.now()
            new_state['updated_at'] = datetime.now()
            
            logger.info(f"새로운 상태 생성 데이터: {new_state}")  # 디버깅용 로그
            collection.insert_one(new_state)
            return new_state
            
    except Exception as e:
        logger.error(f"❌ 상태 조회/생성 중 오류 발생: {e}")
        return { "chat_id": chat_id }


def update_state(chat_id: str, new_values: dict) -> UpdateResult:
    if db is None:
        raise ConnectionError("DB에 연결되지 않았습니다.")

    try:
        collection = db.states
        
        # 기존 문서 확인
        existing_doc = collection.find_one({"chat_id": chat_id})
        logger.info(f"기존 문서 찾기 결과: {existing_doc}")
        
        set_doc = dict(new_values)
        logger.info(f"업데이트할 데이터: {set_doc}")
        
        result = collection.update_one(
            {"chat_id": chat_id},
            {"$set": set_doc,"$currentDate": {"updated_at": True}}
        )
        
        if result.matched_count > 0:
            logger.info(f"✅ 상태가 업데이트 되었습니다. (채팅방 ID: {chat_id}, matched: {result.matched_count}, modified: {result.modified_count})")
            
            # 업데이트 후 상태 확인 및 로그 출력
            updated_doc = collection.find_one({"chat_id": chat_id})
            logger.info(f"업데이트 후 상태: {updated_doc}")
        else:
            logger.warning(f"⚠️ 업데이트할 상태를 찾을 수 없습니다. (채팅방 ID: {chat_id})")
        
        return result

    except Exception as e:
        logger.error(f"❌ 상태 업데이트 중 오류 발생: {e}")
        raise