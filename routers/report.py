from fastapi import APIRouter, Depends, Query, HTTPException, Header
from redis.asyncio import Redis
from core.redis import get_redis
from models.report import TopSlideReport
from services.report_service import get_top_slide_report
from models.common import BaseResponse, success
router = APIRouter(prefix="/report", tags=["report"])

@router.get("/{room_id}/top-slide", response_model=BaseResponse[TopSlideReport],
    summary="질문이 가장 많았던 슬라이드 조회",
    description="roomId에 해당하는 발표에서 **가장 질문이 많았던 슬라이드**와 그 슬라이드의 질문들을 반환합니다.")

async def top_slide(
    room_id: str,
    latest_first: bool = Query(False, description="질문 목록을 최신순으로 정렬"),
    r: Redis = Depends(get_redis),
):
    report = await get_top_slide_report(r, room_id, latest_first=latest_first)

    return success(report)