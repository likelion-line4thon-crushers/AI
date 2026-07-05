from typing import List, Optional, Dict
from models.question_report import QuestionRecord
from core.redis import get_redis


async def list_room_questions(room_id: str, from_ts: Optional[int] = None) -> List[QuestionRecord]:
    # room:{roomId}:questions ZSet에서 전체 질문을 읽어 deleted 제외하고 반환한다.
    # top3 클러스터링 (발표 종료 후 AI 리포트용)에서 사용한다.
    redis = await get_redis()
    zkey = f"room:{room_id}:questions"
    min_score = f"({from_ts}" if from_ts is not None else "-inf"
    max_score = "+inf"

    tuples = await redis.zrangebyscore(zkey, min_score, max_score, withscores=True)
    if not tuples:
        return []

    ids = [qid for qid, _ in tuples]

    # 삭제된 질문 ID Set 조회 (한 번의 Redis 호출로 일괄 처리)
    deleted_ids = await redis.smembers(f"room:{room_id}:questions:deleted")

    pipe = redis.pipeline()
    for qid in ids:
        pipe.hgetall(f"room:{room_id}:question:{qid}")
    hashes: List[Dict[str, str]] = await pipe.execute()

    out: List[QuestionRecord] = []
    for h in hashes:
        if not h:
            continue
        if h.get("id") in deleted_ids:
            continue
        try:
            out.append(
                QuestionRecord(
                    id=h["id"],
                    roomId=h["roomId"],
                    slide=int(h["slide"]),
                    audienceId=h.get("audienceId"),
                    content=h["content"],
                    ts=int(h["ts"]),
                )
            )
        except KeyError:
            continue

    out.sort(key=lambda r: r.ts)
    return out


async def list_completed_questions(room_id: str) -> List[QuestionRecord]:
    # room:{roomId}:questions:completed Set에서 완료된 질문 ID를 읽어 반환한다.
    # AI 리포트 "답변 완료한 질문" 목록에서 사용한다.
    redis = await get_redis()
    completed_ids = await redis.smembers(f"room:{room_id}:questions:completed")
    if not completed_ids:
        return []

    pipe = redis.pipeline()
    for qid in completed_ids:
        pipe.hgetall(f"room:{room_id}:question:{qid}")
    hashes: List[Dict[str, str]] = await pipe.execute()

    out: List[QuestionRecord] = []
    for h in hashes:
        if not h:
            continue
        if h.get("status") != "completed":
            continue
        try:
            out.append(
                QuestionRecord(
                    id=h["id"],
                    roomId=h["roomId"],
                    slide=int(h["slide"]),
                    audienceId=h.get("audienceId"),
                    content=h["content"],
                    ts=int(h["ts"]),
                )
            )
        except KeyError:
            continue

    out.sort(key=lambda r: r.ts)
    return out


async def list_all_questions(room_id: str) -> List[QuestionRecord]:
    # deleted 제외한 전체 질문 반환한다 (active + completed).
    # AI 리포트 "질문 모두 보기"에서 사용한다.
    return await list_room_questions(room_id)
