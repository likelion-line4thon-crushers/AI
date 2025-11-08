from typing import List, Optional
from pydantic import BaseModel

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
