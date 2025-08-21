from datetime import datetime
from .connection import db  
import logging 

logger = logging.getLogger(__name__)

def save_chat_message(chat_id: str, user_message: str, agent_message: str):
    if db is None:
        logger.error("DB에 연결되지 않아 메시지를 저장할 수 없습니다.")
        return None

    try:
        collection = db.chat_logs
        
        message_log = {
            "chat_id": chat_id,
            "user_message": user_message,
            "agent_message": agent_message,
            "timestamp": datetime.now()
        }
        
        result = collection.insert_one(message_log)
        logger.info(f"💬 메시지가 성공적으로 저장되었습니다. (ID: {result.inserted_id})")
        return result.inserted_id

    except Exception as e:
        logger.error(f"❌ 메시지 저장 중 오류가 발생했습니다: {e}")
        return None

def get_chat_history(chat_id: str, limit: int = 10):
    
    if db is None:
        logger.error("DB에 연결되지 않아 기록을 조회할 수 없습니다.")
        return []

    try:
        collection = db.chat_logs
        
        history = list(collection.find({"chat_id": chat_id})
                                 .sort("timestamp", -1)
                                 .limit(limit))
        
        return history[::-1] 

    except Exception as e:
        logger.error(f"❌ 채팅 기록 조회 중 오류가 발생했습니다: {e}")
        return []

def delete_chat_history(chat_id: str): 
    if db is None:
        logger.error("DB에 연결되지 않아 기록을 삭제할 수 없습니다.")
        return 0
    
    try:
        collection = db.chat_logs
        result = collection.delete_many({"chat_id": chat_id})
        deleted_count = result.deleted_count

        logger.info(f"💬 총 {deleted_count}개의 메시지가 삭제되었습니다. (채팅방 ID: {chat_id})")
        return deleted_count
    
    except Exception as e:
        logger.error(f"❌ 채팅 기록 삭제 중 오류가 발생했습니다: {e}")
        return 0