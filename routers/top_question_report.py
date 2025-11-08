from fastapi import APIRouter, Query
from services.question_reader import list_room_questions
from services.top3_service import build_top3
from models.question_report import TopQuestionReportResponse
from models.common import BaseResponse, success

router = APIRouter(prefix="/report", tags=["Report"])

@router.get("/questions/rooms/{room_id}/top3", response_model=BaseResponse[TopQuestionReportResponse],
            summary="TOP3",
            description="지정된 room_id의 질문들을 불러와 의미 유사도를 기반으로 묶은 **TOP3 질문 클러스터**를 반환합니다."
)
async def top3_report(room_id: str):
    questions = await list_room_questions(room_id)
    report = build_top3(questions)
    return success(report)