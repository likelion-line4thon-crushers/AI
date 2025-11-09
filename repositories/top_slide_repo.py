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

async def upsert_top_slide_report_null(db: AsyncSession, room_id: str) -> None:
    """
    슬라이드/질문이 없을 때 NULL 값으로 리포트를 초기화(업서트)
    """
    sql = text("""
        INSERT INTO report (
            room_id,
            popular_question
        )
        VALUES (
            :room_id,
            NULL
        )
        ON DUPLICATE KEY UPDATE
            popular_question = VALUES(popular_question)
    """)

    await db.execute(sql, {"room_id": room_id})
    await db.commit()