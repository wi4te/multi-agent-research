"""SQLAlchemy models for the research assistant."""
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid
from database.connection import Base


class Document(Base):
    """Documents stored in the vector database for RAG."""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    source = Column(String(500))
    embedding = Column(Vector(1536))  # Titan embeddings are 1536-dim
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Document {self.id}: {self.content[:50]}>"


class Conversation(Base):
    """Conversation history for memory."""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
