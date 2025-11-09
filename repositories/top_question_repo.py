import json
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from models.question_report import TopQuestionItem  # 네 모델에 맞게 import

async def upsert_top3_null(db: AsyncSession, room_id: str) -> None:
    """
    질문이 0개일 때: report.top3_questions 를 NULL로 업서트
    """
    sql = text("""
        INSERT INTO report (
            room_id,
            top3question
        )
        VALUES (
            :room_id,
            NULL
        )
        ON DUPLICATE KEY UPDATE
            top3question = VALUES(top3question)
    """)
    await db.execute(sql, {"room_id": room_id})
    await db.commit()


async def update_report_top3(db: AsyncSession, room_id: str, items: List[TopQuestionItem]) -> None:
    """
    TOP3 결과를 report.top3_questions(JSON) 컬럼에 저장
    """
    payload = [it.representative for it in items if it.representative]

    sql = text("""
        UPDATE report
        SET top3question = CAST(:data AS JSON)
        WHERE room_id = :room_id
    """)
    await db.execute(sql, {
        "data": json.dumps(payload, ensure_ascii=False),
        "room_id": room_id,
    })
    await db.commit()
