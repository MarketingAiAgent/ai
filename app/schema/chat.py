from re import S
from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    user_message: str
    chat_id: str 
    company: str
    user_id: Optional[str] = None

class NewChatRequest(BaseModel):
    user_message: str
    company: str
    user_id: Optional[str] = None

class CreatePlanRequest(BaseModel):
    chat_id: str