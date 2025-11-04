import re
from typing import List
from redis.asyncio import Redis
from models.report import Question, TopSlideReport

SLIDE_ZSET_PATTERN = "room:{roomId}:page:*:questions"
QUESTION_HASH_FMT = "room:{roomId}:question:{qid}"

async def _scan_keys(r: Redis, pattern: str, count: int = 200) -> List[str]:
    cursor = 0  # int
    keys: List[str] = []
    while True:
        cursor, chunk = await r.scan(cursor=cursor, match=pattern, count=count)
        keys.extend(chunk)
        if cursor == 0:  # int 비교
            break
    return keys

def _parse_slide_no(zset_key: str) -> int:
    m = re.search(r"page:(\d+):questions$", zset_key)
    return int(m.group(1)) if m else 0

async def get_top_slide_report(
    r: Redis, room_id: str, latest_first: bool = False
) -> TopSlideReport:
    pattern = SLIDE_ZSET_PATTERN.format(roomId=room_id)
    slide_keys = await _scan_keys(r, pattern)
    if not slide_keys:
        return TopSlideReport(roomId=room_id, slide=0, totalQuestions=0, questions=[])

    # 각 슬라이드별 개수 조회
    pipe = r.pipeline()
    for k in slide_keys:
        pipe.zcard(k)
    counts = await pipe.execute()

    max_idx = max(range(len(slide_keys)), key=lambda i: counts[i])
    top_key = slide_keys[max_idx]
    top_count = counts[max_idx]
    slide_no = _parse_slide_no(top_key)

    # question id 목록 (정렬 방향 옵션)
    qids = await (r.zrevrange(top_key, 0, -1) if latest_first else r.zrange(top_key, 0, -1))
    if not qids:
        return TopSlideReport(roomId=room_id, slide=slide_no, totalQuestions=0, questions=[])

    # 상세 해시 벌크 조회
    pipe = r.pipeline()
    for qid in qids:
        pipe.hgetall(QUESTION_HASH_FMT.format(roomId=room_id, qid=qid))
    rows = await pipe.execute()

    questions: List[Question] = []
    for qid, row in zip(qids, rows):
        if not row:  # TTL로 사라진 경우
            continue
        try:
            questions.append(Question(
                id=qid,
                slide=int(row.get("slide", slide_no)),
                content=row.get("content", ""),
                ts=int(row.get("ts", "0")),
                audienceId=row.get("audienceId"),
            ))
        except Exception:
            questions.append(Question(id=qid, slide=slide_no, content=row.get("content", ""), ts=0))

    return TopSlideReport(
        roomId=room_id, slide=slide_no, totalQuestions=top_count, questions=questions
    )
