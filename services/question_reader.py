from typing import List, Optional, Dict
from models.question_report import QuestionRecord
from core.redis import get_redis

async def list_room_questions(room_id: str, from_ts: Optional[int] = None) -> List[QuestionRecord]:
    redis = await get_redis()
    zkey = f"room:{room_id}:questions"
    min_score = f"({from_ts}" if from_ts is not None else "-inf"   # (x: exclusive
    max_score = "+inf"

    tuples = await redis.zrangebyscore(zkey, min_score, max_score, withscores=True)
    if not tuples:
        return []

    ids = [qid for qid, _ in tuples]

    pipe = redis.pipeline()
    for qid in ids:
        pipe.hgetall(f"room:{room_id}:question:{qid}")
    hashes: List[Dict[str, str]] = await pipe.execute()

    out: List[QuestionRecord] = []
    for h in hashes:
        if not h:
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
