from datetime import datetime
from zoneinfo import ZoneInfo
from .connection import db  
from pymongo.results import InsertManyResult, UpdateResult, InsertOneResult, DeleteResult
from pymongo import DESCENDING
from typing import Optional
import logging 
import uuid

logger = logging.getLogger(__name__)

def crete_chat(user_id: str, title:str):
    if db is None:
        logger.error("DB에 연결되지 않아 메시지를 저장할 수 없습니다.")
        return None
    
    try:
        collection = db.chats
        now = datetime.now(ZoneInfo("Asia/Seoul"))

        chat_data = { 
            "chat_id": uuid.uuid4().hex,
            "user_id": user_id,
            "title": title,
            "created_at": now,
            "last_updated": now,
            "message_ids": []
        }

        result: InsertOneResult = collection.insert_one(chat_data)

        if result.inserted_id:
            new_chat_id = chat_data["chat_id"]
            logger.info(f"✅ Successfully created a new chat for user '{user_id}' with chat_id: {new_chat_id}")
            return new_chat_id

        else:
            logger.error(f"Failed to create a chat for user '{user_id}'.")
            return None

    except Exception as e:
        logger.error(f"❌ An error occurred while creating a chat for user '{user_id}': {e}")
        return None
        
def save_chat_message(chat_id: str, user_message: str, agent_message: str, graph_data, plan_data: Optional[str]):
    if db is None:
        logger.error("DB에 연결되지 않아 메시지를 저장할 수 없습니다.")
        return None

    try:
        print("graph_data: ", graph_data)
        
        collection = db.messages
        base_time = datetime.now(ZoneInfo("Asia/Seoul"))
        
        # 유저 메시지 먼저 저장 (더 이른 타임스탬프)
        user_message_data = {
            "message_id": uuid.uuid4().hex,
            "chat_id": chat_id,
            "speaker": "user",
            "timestamp": base_time,
            "content": user_message,
            "graph_data": None,
            "plan_data": None
        }
        
        user_result = collection.insert_one(user_message_data)
        if not user_result.acknowledged:
            logger.error(f"Failed to save user message for chat_id '{chat_id}'.")
            return False
            
        # AI 메시지는 약간의 시간 간격을 두고 저장 (더 늦은 타임스탬프)
        ai_message_data = {
            "message_id": uuid.uuid4().hex,
            "chat_id": chat_id,
            "speaker": "ai",
            "timestamp": base_time.replace(microsecond=base_time.microsecond + 1000),  # 1ms 후
            "content": agent_message,
            "graph_data": graph_data,
            "plan_data": plan_data
        }
        
        ai_result = collection.insert_one(ai_message_data)
        if not ai_result.acknowledged:
            logger.error(f"Failed to save AI message for chat_id '{chat_id}'.")
            return False

        inserted_message_ids = [user_message_data["message_id"], ai_message_data["message_id"]]

        chats_collection = db.chats
        update_result: UpdateResult = chats_collection.update_one(
            {"chat_id": chat_id},
            {
                "$push": {
                    "message_ids": {
                        "$each": inserted_message_ids
                    }
                },
                "$set": { 
                    "last_updated": base_time 
                }
            },
            upsert=True 
        )

        if update_result.matched_count == 0 and update_result.upserted_id is None:
             logger.error(f"Failed to update or upsert chat document for chat_id '{chat_id}'.")
             return False

        return True

    except Exception as e:
        logger.error(f"❌ An error occurred while saving messages for chat_id '{chat_id}': {e}")
        return None

def get_chat_history(chat_id: str, limit: int = 10):
    
    if db is None:
        logger.error("DB에 연결되지 않아 기록을 조회할 수 없습니다.")
        return []

    try:
        collection = db.messages
        
        messages_cursor = collection.find(
            {"chat_id": chat_id}
        ).sort("timestamp", DESCENDING).limit(limit)

        recent_messages = list(messages_cursor)
        recent_messages.reverse()

        logger.info(f"✅ Fetched {len(recent_messages)} messages for chat_id '{chat_id}'.")
        return recent_messages


    except Exception as e:
        logger.error(f"❌ 채팅 기록 조회 중 오류가 발생했습니다: {e}")
        return []

def delete_chat_history(chat_id: str): 
    if db is None:
        logger.error("DB에 연결되지 않아 기록을 삭제할 수 없습니다.")
        return 0
    
    try:
        messages_collection = db.messages
        chats_collection = db.chats

        delete_messages_result: DeleteResult = messages_collection.delete_many(
            {"chat_id": chat_id}
        )

        delete_chat_result: DeleteResult = chats_collection.delete_one(
            {"chat_id": chat_id}
        )

        logger.info(
            f"✅ Deletion complete for chat_id '{chat_id}': "
            f"{delete_messages_result.deleted_count} messages and "
            f"{delete_chat_result.deleted_count} chat room deleted."
        )

        return True
    
    except Exception as e:
        logger.error(f"❌ 채팅 기록 삭제 중 오류가 발생했습니다: {e}")
        return False