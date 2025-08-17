from datetime import datetime
from .connection import db  

def save_chat_message(thread_id: str, user_message: str, agent_message: str, state:dict):
    if db is None:
        print("DB에 연결되지 않아 메시지를 저장할 수 없습니다.")
        return None

    try:
        collection = db.chat_logs
        
        message_log = {
            "thread_id": thread_id,
            "user_message": user_message,
            "agent_message": agent_message,
            "timestamp": datetime.now()
        }
        
        result = collection.insert_one(message_log)
        print(f"💬 메시지가 성공적으로 저장되었습니다. (ID: {result.inserted_id})")
        return result.inserted_id

    except Exception as e:
        print(f"❌ 메시지 저장 중 오류가 발생했습니다: {e}")
        return None

def get_chat_history(thread_id: str, limit: int = 10):
    
    if db is None:
        print("DB에 연결되지 않아 기록을 조회할 수 없습니다.")
        return []

    try:
        collection = db.chat_logs
        
        history = list(collection.find({"thread_id": thread_id})
                                 .sort("timestamp", -1)
                                 .limit(limit))
        
        return history[::-1] 

    except Exception as e:
        print(f"❌ 채팅 기록 조회 중 오류가 발생했습니다: {e}")
        return []