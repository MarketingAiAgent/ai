import os
from pymongo import MongoClient
from app.core.config import settings

def get_db_client():
    connection_string = settings.COSMOS_DB_CONNECTION_STRING

    if not connection_string:
        raise ValueError("COSMOS_DB_CONNECTION_STRING 환경 변수를 설정해주세요.")

    try:
        client = MongoClient(connection_string)
        client.admin.command('ping')
        print("✅ MongoDB 연결에 성공했습니다.")
        
        return client

    except Exception as e:
        print(f"❌ MongoDB 연결에 실패했습니다: {e}")
        return None

db_client = get_db_client()

def get_database(db_name):

    if db_client:
        return db_client[db_name]
    return None

db = get_database("minti")