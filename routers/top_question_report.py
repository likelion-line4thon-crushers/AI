from fastapi import APIRouter, Query
from services.question_reader import list_room_questions
from services.top3_service import build_top3
from models.common import BaseResponse, success
from typing import List
from models.question_report import TopQuestionReportResponse

router = APIRouter(prefix="/report", tags=["Report"])

@router.get("/questions/rooms/{room_id}/top3", response_model=TopQuestionReportResponse)
async def top3_report(room_id: str):
    questions = await list_room_questions(room_id)
    return build_top3(questions)