from pydantic import BaseModel
from typing import List, Optional

class QuestionRecord(BaseModel):
    id: str
    roomId: str
    slide: int
    audienceId: Optional[str] = None
    content: str
    ts: int

class TopQuestionItem(BaseModel):
    representative: str          # 대표 문구
    count: int                   # 그룹 내 질문 수
    questionIds: List[str]       # 묶인 질문 ID들
    slides: List[int]            # 등장한 슬라이드(중복 제거 후 정렬)
    samples: List[str]           # 샘플 3개

class TopQuestionReportResponse(BaseModel):
    roomId: str
    totalQuestions: int          # 전체 질문 수 (모든 슬라이드 합계)
    uniqueGroups: int            # 유사도 기준으로 묶인 고유 그룹 개수
    top3: List[TopQuestionItem]  # 가장 많이 언급된 상위 3개 질문 그룹 리스트
