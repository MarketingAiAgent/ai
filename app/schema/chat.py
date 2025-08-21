from re import S
from ._base import CamelCaseModel
from typing import Optional

class ChatRequest(CamelCaseModel):
    user_message: str
    chat_id: str 
    company: str
    user_id: Optional[str] = None

class NewChatRequest(CamelCaseModel):
    user_message: str
    company: str
    user_id: Optional[str] = None

class CreatePlanRequest(CamelCaseModel):
    chat_id: str