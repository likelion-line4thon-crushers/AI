from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from services.question_reader import list_room_questions, list_completed_questions, list_all_questions
from core.db import get_db
from services.top3_service import build_top3
from services.incremental_cluster_service import add_question_to_clusters
from models.question_report import TopQuestionReportResponse, QuestionRecord
from models.cluster import QuestionInput, ClusterReportResponse
from models.common import BaseResponse, success

router = APIRouter(prefix="/report", tags=["Report"])


@router.get("/questions/rooms/{room_id}/top3", response_model=BaseResponse[TopQuestionReportResponse],
            summary="TOP3",
            description="지정된 room_id의 질문들을 불러와 의미 유사도를 기반으로 묶은 **TOP3 질문 클러스터**를 반환합니다."
)
async def top3_report(room_id: str, db: AsyncSession = Depends(get_db)):
    # deleted 제외한 전체 질문으로 클러스터링 후 상위 3개 반환 (발표 종료 후 AI 리포트용)
    questions = await list_room_questions(room_id)
    report = await build_top3(room_id, questions, db)
    return success(report)


@router.get(
    "/questions/rooms/{room_id}/completed",
    response_model=BaseResponse[List[QuestionRecord]],
    summary="답변 완료한 질문 목록",
    description="발표자가 완료 처리한 질문 목록을 반환합니다. AI 리포트 '답변 완료한 질문' 탭에서 사용합니다.",
)
async def completed_questions(room_id: str):
    # room:{roomId}:questions:completed Set 기반으로 완료 질문만 반환
    questions = await list_completed_questions(room_id)
    return success(questions)


@router.get(
    "/questions/rooms/{room_id}/all",
    response_model=BaseResponse[List[QuestionRecord]],
    summary="전체 질문 목록 (deleted 제외)",
    description="삭제된 질문을 제외한 전체 질문 목록을 반환합니다. AI 리포트 '질문 모두 보기' 탭에서 사용합니다.",
)
async def all_questions(room_id: str):
    # active + completed 질문 모두 반환, deleted만 제외
    questions = await list_all_questions(room_id)
    return success(questions)


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
