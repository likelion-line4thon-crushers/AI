from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from services.question_reader import list_room_questions
from core.db import get_db
from services.top3_service import build_top3
from services.incremental_cluster_service import add_question_to_clusters
from models.question_report import TopQuestionReportResponse
from models.cluster import QuestionInput, ClusterReportResponse
from models.common import BaseResponse, success

router = APIRouter(prefix="/report", tags=["Report"])

@router.get("/questions/rooms/{room_id}/top3", response_model=BaseResponse[TopQuestionReportResponse],
            summary="TOP3",
            description="지정된 room_id의 질문들을 불러와 의미 유사도를 기반으로 묶은 **TOP3 질문 클러스터**를 반환합니다."
)
async def top3_report(room_id: str, db: AsyncSession = Depends(get_db),):
    questions = await list_room_questions(room_id)
    report = await build_top3(room_id,questions, db)
    return success(report)


@router.post(
    "/questions/rooms/{room_id}/clusters/incremental",
    response_model=BaseResponse[ClusterReportResponse],
    summary="증분 클러스터링",
    description="새 질문 1개를 기존 클러스터 상태에 증분 추가하고 전체 클러스터 결과를 반환합니다.",
)
async def incremental_cluster(room_id: str, question: QuestionInput):
    # Spring Boot가 질문 저장 직후 호출한다.
    # Redis에 누적된 클러스터 상태에 새 질문을 끼워넣고 전체 결과를 반환한다.
    result = await add_question_to_clusters(room_id, question)
    return success(result)