from fastapi import APIRouter, Depends, Query, HTTPException, Header
from redis.asyncio import Redis
from core.redis import get_redis
from models.report import TopSlideReport
from services.report_service import get_top_slide_report
from config.settings import settings

router = APIRouter(prefix="/report", tags=["report"])

def _auth(x_api_key: str | None = Header(default=None)):
    # 옵션: settings.API_KEY 설정 시 헤더로 검증
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@router.get("/{room_id}/top-slide", response_model=TopSlideReport)
async def top_slide(
    room_id: str,
    latest_first: bool = Query(False, description="질문 목록을 최신순으로 정렬"),
    r: Redis = Depends(get_redis),
):
    report = await get_top_slide_report(r, room_id, latest_first=latest_first)
    if report.totalQuestions == 0 and report.slide == 0:
        raise HTTPException(status_code=404, detail="No questions found for this room")
    return report