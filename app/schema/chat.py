from re import S
from ._base import CamelCaseModel
from typing import Optional

class ChatRequest(CamelCaseModel):
    user_message: str
    chat_id: str 
    company: str
    user_id: Optional[str] = None

class NewChatRequest(CamelCaseModel):
    company: str
    user_id: Optional[str] = None

class CreatePlanRequest(CamelCaseModel):
    chat_id: str
    user_id: str
    company: str

class CreateBrandPlanResponse(CamelCaseModel):
    planId: str
    title: str
    mainBanner: str
    couponSection: str
    productSection: str
    eventNotes: str

class CreateCategoryPlanResponse(CamelCaseModel):
    planId: str
    title: str
    mainBanner: str
    section1: str
    section2: str
    section3: str