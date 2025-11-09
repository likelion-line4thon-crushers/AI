import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from models.max_slide_report import TopSlideReport

async def update_report_popular_question(db: AsyncSession, rpt: TopSlideReport) -> None:
    """TopSlideReport 결과를 report.popular_question 컬럼에 JSON으로 저장"""
    payload = {
        "slide": rpt.slide,
        "questions": [q.model_dump() for q in rpt.questions],
        "summary": rpt.summary,
    }

    sql = text("""
        UPDATE report
        SET popular_question = CAST(:data AS JSON)
        WHERE room_id = :room_id
    """)

    await db.execute(sql, {
        "data": json.dumps(payload, ensure_ascii=False),
        "room_id": rpt.roomId,
    })
    await db.commit()
