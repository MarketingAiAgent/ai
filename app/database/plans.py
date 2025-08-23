from .connection import db  
from pymongo.results import InsertManyResult, UpdateResult, InsertOneResult, DeleteResult
from pymongo import DESCENDING
import logging 
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def create_plan(plan_id: str, company: str, target_type: str, plan_content: dict):
    if db is None:
        logger.error("DB에 연결되지 않아 메시지를 저장할 수 없습니다.")
        return None
    
    try:
        collection = db.plans
        now = datetime.now(ZoneInfo("Asia/Seoul"))

        plan_data = {
            "plan_id": plan_id,
            "company": company,
            "created_at": now,
            "last_updated": now,
            "target_type": target_type,
            "plan_content": plan_content,
            "share": False,
            "url": None
        }
        
        result: InsertOneResult = collection.insert_one(plan_data)
        
        if result.inserted_id:
            logger.info(f"✅ Successfully created a new plan for company '{company}' with plan_id: {plan_id}")
            return result.inserted_id
        
        else:
            logger.error(f"Failed to create a plan for company '{company}'.")
            return None
        
    except Exception as e:
        logger.error(f"Error creating plan for company '{company}': {e}")
        return None