import datetime
from pydantic import BaseModel, ConfigDict

class MessageCreate(BaseModel):
    message_id: str
    chatroom_id: str
    type: str
    content: str

class Message(BaseModel):
    model_config = ConfigDict(from_attributes=True) 

    message_id: str
    chatroom_id: str
    type: str | None = None
    content: str | None = None
    created_at: datetime.datetime