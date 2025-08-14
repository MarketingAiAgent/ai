from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class Chatroom(Base):
    __tablename__ = "chatroom"
    chatroom_id = Column(String(255), primary_key=True)
    action_task = Column(JSON)
    messages = relationship("Message", back_populates="chatroom")

class ChatMessage(Base):
    __tablename__ = "messages"
    message_id = Column(String(255), primary_key=True)
    chatroom_id = Column(String(255), ForeignKey("chatroom.chatroom_id"), nullable=False)
    type = Column(String(50))
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    chatroom = relationship("Chatroom", back_populates="messages")

class DbInfo(Base):
    __tablename__ = "db_info"
    company_name = Column(String(255), primary_key=True)
    connection_info = Column(Text, nullable=False)