import json
import logging
import numpy as np
from typing import List, Dict

from models.cluster import QuestionInput, ClusterItem, ClusterReportResponse
from services import text_sim as TS
from core.redis import get_redis
from exception.errors import AppException, ReportErrorCode
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMB_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
EMB_THRESHOLD = 0.45       # 코사인 유사도 임계값: 이 이상이면 같은 클러스터로 판단
NGRAM = 2                   # 문자 n-gram 크기 (simhash용)
HAMMING_THRESHOLD = 4       # simhash 해밍 거리 임계값: 이 이하이면 같은 클러스터로 판단
JACCARD_FALLBACK = 0.60     # 코사인/해밍 둘 다 실패 시 자카드 유사도 기준

_model = SentenceTransformer(EMB_MODEL)


def _embed(text: str) -> np.ndarray:
    # 텍스트를 정규화한 뒤 한국어 SBERT로 임베딩 (코사인 정규화 포함)
    try:
        return _model.encode(TS.normalize(text), normalize_embeddings=True)
    except Exception as e:
        logger.error(f"[임베딩] '{text[:20]}' 처리 실패: {e}")
        raise AppException(ReportErrorCode.EMBED_ERROR, detail=str(e))


def _clusters_key(room_id: str) -> str:
    # Redis에 클러스터 상태를 저장하는 키
    return f"room:{room_id}:clusters"


def _total_key(room_id: str) -> str:
    # Redis에서 전체 질문 수를 읽는 키 (Spring Boot에서 증가시킴)
    return f"room:{room_id}:questionCount"


async def add_question_to_clusters(room_id: str, question: QuestionInput) -> ClusterReportResponse:
    # 새 질문 1개를 기존 클러스터 상태에 증분 추가하고 전체 클러스터 결과를 반환한다.
    # 판단 우선순위: 코사인 유사도 → 해밍 거리 → 자카드 유사도 → 신규 클러스터 생성
    # 클러스터 상태는 Redis(room:{roomId}:clusters)에 JSON으로 저장/갱신된다.
    redis = await get_redis()

    try:
        # 새 질문 전처리: 임베딩, n-gram, simhash 계산
        emb = _embed(question.content)
        norm = TS.normalize(question.content)
        sh = TS.char_ngrams(norm, NGRAM)
        simh = TS.simhash64(sh)

        # 기존 클러스터 상태 로드
        raw = await redis.get(_clusters_key(room_id))
        clusters: List[Dict] = json.loads(raw) if raw else []

        joined = False
        best_cos_idx = -1
        best_cos = -1.0
        best_d_idx = -1
        best_d = 1 << 30

        # 전체 클러스터와 코사인/해밍 유사도 비교해서 가장 가까운 클러스터 탐색
        for i, c in enumerate(clusters):
            cent_emb = np.array(c["centroid_emb"], dtype=np.float32)
            cos = float(emb @ cent_emb)
            if cos > best_cos:
                best_cos = cos
                best_cos_idx = i

            d = TS.hamming(simh, c["simhash"])
            if d < best_d:
                best_d = d
                best_d_idx = i

        if best_cos_idx >= 0 and best_cos >= EMB_THRESHOLD:
            # 코사인 유사도 기준 합류: 중심 벡터를 새 질문으로 갱신 (centroid drift 발생 지점)
            c = clusters[best_cos_idx]
            c["member_ids"].append(question.id)
            c["slides"] = sorted(set(c["slides"] + [question.slide]))
            if len(c["samples"]) < 3:
                c["samples"].append(question.content)
            c["count"] += 1
            c["centroid_emb"] = emb.tolist()
            joined = True

        elif best_d_idx >= 0 and best_d <= HAMMING_THRESHOLD:
            # 해밍 거리 기준 합류: 문자 패턴이 비슷한 경우
            c = clusters[best_d_idx]
            c["member_ids"].append(question.id)
            c["slides"] = sorted(set(c["slides"] + [question.slide]))
            if len(c["samples"]) < 3:
                c["samples"].append(question.content)
            c["count"] += 1
            joined = True

        if not joined:
            # 자카드 유사도 fallback: n-gram 집합 겹침 기준
            for c in clusters:
                c_sh = set(c["ngrams"])
                jac = TS.jaccard(sh, c_sh)
                if jac >= JACCARD_FALLBACK:
                    c["member_ids"].append(question.id)
                    c["slides"] = sorted(set(c["slides"] + [question.slide]))
                    if len(c["samples"]) < 3:
                        c["samples"].append(question.content)
                    c["count"] += 1
                    joined = True
                    break

            if not joined:
                # 어느 클러스터에도 속하지 않으면 신규 클러스터 생성
                clusters.append({
                    "representative": question.content,
                    "centroid_emb": emb.tolist(),
                    "simhash": simh,
                    "ngrams": list(sh),
                    "member_ids": [question.id],
                    "slides": [question.slide],
                    "samples": [question.content],
                    "count": 1,
                })

        # 갱신된 클러스터 상태를 Redis에 저장
        await redis.set(_clusters_key(room_id), json.dumps(clusters))

        # 전체 질문 수는 Spring Boot가 관리하는 카운터에서 읽음
        total_raw = await redis.get(_total_key(room_id))
        total = int(total_raw) if total_raw else 0

        # count 내림차순 정렬 후 응답 변환
        clusters_sorted = sorted(clusters, key=lambda c: c["count"], reverse=True)
        items = [
            ClusterItem(
                representative=c["representative"],
                count=c["count"],
                questionIds=c["member_ids"],
                slides=c["slides"],
                samples=c["samples"],
            )
            for c in clusters_sorted
        ]

        logger.info(f"[IncrementalCluster] roomId={room_id}, 총 {len(clusters)}개 그룹")

        return ClusterReportResponse(
            roomId=room_id,
            totalQuestions=total,
            uniqueGroups=len(clusters),
            clusters=items,
        )

    except AppException:
        raise
    except Exception as e:
        logger.exception(f"[IncrementalCluster] 알 수 없는 오류: {e}")
        raise AppException(ReportErrorCode.UNKNOWN, detail=str(e))
