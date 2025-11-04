import re
import logging
from typing import List
from redis.asyncio import Redis
from models.report import Question, TopSlideReport

logger = logging.getLogger(__name__)  # 모듈 로거 등록

SLIDE_ZSET_PATTERN = "room:{roomId}:page:*:questions"
QUESTION_HASH_FMT = "room:{roomId}:question:{qid}"

##  Redis에서 특정 패턴(room:{roomId}:page:*:questions)에 맞는 모든 키를 스캔하는 함수
##     - Redis의 SCAN 명령을 사용해서 슬라이드별 질문 목록 키(ZSET)들을 찾음
##     - 한 번에 너무 많은 키를 읽지 않기 위해 count 단위로 반복 스캔
##     - 결과: ["room:...:page:1:questions", "room:...:page:2:questions", ...]
async def _scan_keys(r: Redis, pattern: str, count: int = 200) -> List[str]:
    cursor = 0  # int
    keys: List[str] = []
    logger.debug(f"[SCAN] Redis 키 스캔 시작: pattern={pattern}")
    while True:
        cursor, chunk = await r.scan(cursor=cursor, match=pattern, count=count)
        keys.extend(chunk)
        logger.debug(f"[SCAN] {len(chunk)}개의 키 조회됨 (cursor={cursor})")
        if cursor == 0:  # int 비교
            break
    logger.info(f"[SCAN] 총 {len(keys)}개의 슬라이드 키 발견")
    return keys

## 🔹 Redis 키 문자열에서 슬라이드 번호(page:X)를 추출하는 함수
##     - 예: "room:abc:page:3:questions" → 3
##     - 없으면 기본값 0 반환
def _parse_slide_no(zset_key: str) -> int:
    m = re.search(r"page:(\d+):questions$", zset_key)
    slide_no = int(m.group(1)) if m else 0
    logger.debug(f"[PARSE] 키에서 슬라이드 번호 추출: {slide_no} ({zset_key})")
    return slide_no

##   실제 "최다 질문 슬라이드 리포트"를 생성하는 핵심 함수
##     1️. 해당 room_id의 모든 슬라이드 키를 스캔
##     2️. 각 슬라이드별 질문 개수를 ZCARD로 계산
##     3️. 질문이 가장 많은 슬라이드(top slide)를 선택
##     4️. 해당 슬라이드의 질문 ID 목록(ZRANGE or ZREVRANGE) 조회
##     5️. 각 질문의 상세 정보(HGETALL) 벌크 조회
##     6️. Question 모델 리스트로 변환 후 TopSlideReport로 반환
async def get_top_slide_report(
    r: Redis, room_id: str, latest_first: bool = False
) -> TopSlideReport:
    logger.info(f"[리포트] room={room_id}의 최다 질문 슬라이드 리포트 생성 시작")
    pattern = SLIDE_ZSET_PATTERN.format(roomId=room_id)
    slide_keys = await _scan_keys(r, pattern)
    if not slide_keys:
        logger.warning(f"[리포트] 해당 room({room_id})에는 질문 데이터가 없습니다.")
        return TopSlideReport(roomId=room_id, slide=0, totalQuestions=0, questions=[])

    # 각 슬라이드별 개수 조회
    pipe = r.pipeline()
    for k in slide_keys:
        pipe.zcard(k)
    counts = await pipe.execute()

    logger.debug(f"[리포트] 각 슬라이드별 질문 수: {counts}")

    max_idx = max(range(len(slide_keys)), key=lambda i: counts[i])
    top_key = slide_keys[max_idx]
    top_count = counts[max_idx]
    slide_no = _parse_slide_no(top_key)

    logger.info(f"[리포트] 최다 질문 슬라이드: {slide_no} (질문 {top_count}개)")

    # question id 목록 (정렬 방향 옵션)
    qids = await (r.zrevrange(top_key, 0, -1) if latest_first else r.zrange(top_key, 0, -1))
    logger.debug(f"[리포트] 슬라이드 {slide_no}에서 {len(qids)}개의 질문 ID 조회 완료")

    if not qids:
        logger.warning(f"[리포트] 슬라이드 {slide_no}에는 질문이 없습니다.")
        return TopSlideReport(roomId=room_id, slide=slide_no, totalQuestions=0, questions=[])

    # 상세 해시 벌크 조회
    pipe = r.pipeline()
    for qid in qids:
        pipe.hgetall(QUESTION_HASH_FMT.format(roomId=room_id, qid=qid))
    rows = await pipe.execute()

    questions: List[Question] = []
    for qid, row in zip(qids, rows):
        if not row:  # TTL로 사라진 경우
            logger.debug(f"[리포트] 만료된 질문(qid={qid}) 건너뜀")
            continue
        try:
            questions.append(Question(
                id=qid,
                slide=int(row.get("slide", slide_no)),
                content=row.get("content", ""),
                ts=int(row.get("ts", "0")),
                audienceId=row.get("audienceId"),
            ))
        except Exception as e:
            logger.error(f"[리포트] 질문(qid={qid}) 파싱 중 오류 발생: {e}")
            questions.append(Question(id=qid, slide=slide_no, content=row.get("content", ""), ts=0))

    logger.info(f"[리포트] room={room_id} 리포트 생성 완료 (총 {len(questions)}개의 질문 포함)")

    return TopSlideReport(
        roomId=room_id, slide=slide_no, totalQuestions=top_count, questions=questions
    )
