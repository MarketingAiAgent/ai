from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    user_message: str
    thread_id: str 
    org_id: str
    db_connection_string: str
    user_id: Optional[str] = None