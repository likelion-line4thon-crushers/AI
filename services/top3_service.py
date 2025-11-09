from typing import List, Set, Dict
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from models.question_report import QuestionRecord, TopQuestionItem, TopQuestionReportResponse
from . import text_sim as TS
from exception.errors import AppException, ReportErrorCode  # [추가]
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession  # ← DB 세션 타입힌트
from repositories.top_question_repo import (     # ← 너가 방금 만든 레포지토리
    upsert_top3_null,                            #     0개일 때 NULL 업서트
    update_report_top3,                          #     TOP3를 JSON으로 저장
)

logger = logging.getLogger(__name__)

EMB_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"  # 한국어 SBERT
EMB_THRESHOLD = 0.45  # 의미가 비슷하다고 판단할 기준

# 성능 보완용 fallback
NGRAM = 2
HAMMING_THRESHOLD = 4
JACCARD_FALLBACK = 0.60
BUCKET_BITS = 14

_model = SentenceTransformer(EMB_MODEL)

def _embed(text: str) -> np.ndarray:
    # 문장을 임베딩 (코사인 정규화 포함)
    try:
        return _model.encode(TS.normalize(text), normalize_embeddings=True)
    except Exception as e:
        logger.error(f"[임베딩] '{text[:20]}...' 처리 실패: {e}")
        raise AppException(ReportErrorCode.EMBED_ERROR, detail=str(e))  # [추가]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    # 코사인 유사도 계산
    try:
        return float(a @ b)
    except Exception as e:
        logger.error(f"[코사인] 계산 실패: {e}")
        raise AppException(ReportErrorCode.CALC_ERROR, detail=str(e))  # [추가]

# 내부 클래스
class _Q:
    __slots__ = ("q", "norm", "sh", "simh", "emb")

    def __init__(self, q: QuestionRecord):
        try:
            self.q = q
            self.norm = TS.normalize(q.content)
            self.sh: Set[str] = TS.char_ngrams(self.norm, NGRAM)
            self.simh = TS.simhash64(self.sh)
            self.emb = _embed(q.content)
        except Exception as e:
            logger.error(f"[질문전처리] id={getattr(q, 'id', '?')} 처리 중 오류: {e}")
            raise AppException(ReportErrorCode.PREPROCESS_ERROR, detail=str(e))  # [추가]

class _Cluster:
    __slots__ = ("centroid", "members", "rep", "slides", "ids", "samples", "cent_emb")
    def __init__(self, first: _Q):
        self.centroid = first.simh
        self.members: List[_Q] = [first]
        self.rep = first.q.content
        self.slides = {first.q.slide}
        self.ids = {first.q.id}
        self.samples = [first.q.content]
        self.cent_emb = first.emb

# 메인 로직
async def build_top3(room_id: str, questions: List[QuestionRecord], db: AsyncSession) -> TopQuestionReportResponse:
    # 질문 리스트를 의미/문자 기반으로 클러스터링하여 상위 3개 그룹 추출
    try:
        if not questions:
            logger.info("[Top3] 입력된 질문이 없습니다.")
            await upsert_top3_null(db, room_id)
            return TopQuestionReportResponse(roomId=room_id,totalQuestions=0, uniqueGroups=0, top3=[])

        items = [_Q(q) for q in questions]

        # LSH-ish 버킷 (성능)
        buckets: Dict[int, List[_Q]] = {}
        for it in items:
            key = it.simh >> (64 - BUCKET_BITS)
            buckets.setdefault(key, []).append(it)

        clusters: List[_Cluster] = []

        for bucket in buckets.values():
            for cur in bucket:
                best = None
                best_cos = -1.0
                best_d = 1 << 30

                for c in clusters:
                    cos = _cos(cur.emb, c.cent_emb)
                    if cos > best_cos:
                        best_cos = cos
                        best = c
                    d = TS.hamming(cur.simh, c.centroid)
                    if d < best_d:
                        best_d = d

                joined = False
                if best and best_cos >= EMB_THRESHOLD:
                    # 의미 유사도 기준으로 합류
                    c = best
                    c.members.append(cur)
                    c.slides.add(cur.q.slide)
                    c.ids.add(cur.q.id)
                    c.samples.append(cur.q.content)
                    c.cent_emb = cur.emb
                    joined = True

                elif best and best_d <= HAMMING_THRESHOLD:
                    # 해밍 거리 기준으로 합류
                    c = best
                    c.members.append(cur)
                    c.slides.add(cur.q.slide)
                    c.ids.add(cur.q.id)
                    c.samples.append(cur.q.content)
                    joined = True

                else:
                    # 자카드 fallback
                    for c in clusters:
                        if not c.members:
                            continue
                        jac = TS.jaccard(cur.sh, c.members[0].sh)
                        if jac >= JACCARD_FALLBACK:
                            c.members.append(cur)
                            c.slides.add(cur.q.slide)
                            c.ids.add(cur.q.id)
                            if len(c.samples) < 3:
                                c.samples.append(cur.q.content)
                            joined = True
                            break

                    if not joined:
                        clusters.append(_Cluster(cur))

        clusters.sort(
            key=lambda c: (len(c.members), max(m.q.ts for m in c.members)),
            reverse=True,
        )

        top3 = [
            TopQuestionItem(
                representative=c.rep,
                count=len(c.members),
                questionIds=list(c.ids),
                slides=sorted(c.slides),
                samples=c.samples,
            )
            for c in clusters[:3]
        ]

        logger.info(f"[Top3] 총 {len(clusters)}개의 그룹 중 상위 3개 반환")

        await update_report_top3(db, room_id, top3)

        return TopQuestionReportResponse(
            roomId=room_id,
            totalQuestions=len(questions),
            uniqueGroups=len(clusters),
            top3=top3,
        )

    except AppException:
        raise

    except RedisError as e:
        logger.error(f"[Top3] Redis 오류: {e}")
        raise AppException(ReportErrorCode.REDIS_ERROR, detail=str(e))

    except Exception as e:
        logger.exception(f"[Top3] 알 수 없는 오류: {e}")
        raise AppException(ReportErrorCode.UNKNOWN, detail=str(e))
