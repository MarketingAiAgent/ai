from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.schema.database import *
from app.database.models import *

async def create_message(db: AsyncSession, message: MessageCreate) -> ChatMessage:
    new_message = ChatMessage(**message.model_dump())
    
    db.add(new_message)
    await db.commit()
    await db.refresh(new_message)
    
    return new_message


async def get_messages_by_chatroom(db: AsyncSession, chatroom_id: str) -> list[ChatMessage]:
    query = select(ChatMessage).where(ChatMessage.chatroom_id == chatroom_id)
    result = await db.execute(query)
    
    return result.scalars().all()