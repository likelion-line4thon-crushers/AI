from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import BigInteger, Integer, String, JSON, Text, text, DateTime

class Base(DeclarativeBase):
    """SQLAlchemy Declarative Base (모든 ORM 모델의 부모)"""
    pass

class Question(BaseModel):
    id: str
    slide: int
    content: str
    ts: int
    audienceId: Optional[str] = None

class TopSlideReport(BaseModel):
    roomId: str
    slide: int
    totalQuestions: int
    questions: List[Question]
    summary: Optional[str] = None

class AiTopSlideReport(Base):
    __tablename__ = "report"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    room_id: Mapped[str] = mapped_column(String(64))
    popular_question: Mapped[Optional[dict]] = mapped_column(JSON)
