import json
import logging
import numpy as np
from typing import List, Dict, Set

from models.cluster import QuestionInput, ClusterItem, ClusterQuestionItem, ClusterReportResponse
from models.training_data import TrainingData
from services import text_sim as TS
from core.redis import get_redis
from core.db import async_session_factory
from exception.errors import AppException, ReportErrorCode
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMB_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
EMB_THRESHOLD = 0.60       # 코사인 유사도 임계값: 이 이상이면 같은 클러스터로 판단
NGRAM = 2                   # 문자 n-gram 크기 (simhash용)
HAMMING_THRESHOLD = 4       # simhash 해밍 거리 임계값: 이 이하이면 같은 클러스터로 판단
JACCARD_FALLBACK = 0.60     # 코사인/해밍 둘 다 실패 시 자카드 유사도 기준

_model = SentenceTransformer(EMB_MODEL)


def _embed(text: str) -> np.ndarray:
    # 정규화 → 형태소 기반 키워드 추출 → SBERT 임베딩 (코사인 정규화 포함)
    try:
        preprocessed = TS.extract_keywords(TS.normalize(text))
        return _model.encode(preprocessed, normalize_embeddings=True)
    except Exception as e:
        logger.error(f"[임베딩] '{text[:20]}' 처리 실패: {e}")
        raise AppException(ReportErrorCode.EMBED_ERROR, detail=str(e))


def _clusters_key(room_id: str) -> str:
    # Redis에 클러스터 상태를 저장하는 키
    return f"room:{room_id}:clusters"


def _total_key(room_id: str) -> str:
    # Redis에서 전체 질문 수를 읽는 키 (Spring Boot에서 증가시킴)
    return f"room:{room_id}:questionCount"


def _question_key(room_id: str, question_id: str) -> str:
    return f"room:{room_id}:question:{question_id}"


def _dedupe(ids: List[str]) -> List[str]:
    seen = set()
    out = []
    for qid in ids:
        if not qid or qid in seen:
            continue
        seen.add(qid)
        out.append(qid)
    return out


async def _inactive_ids(room_id: str) -> Set[str]:
    redis = await get_redis()
    deleted_ids = await redis.smembers(f"room:{room_id}:questions:deleted")
    completed_ids = await redis.smembers(f"room:{room_id}:questions:completed")
    return set(deleted_ids) | set(completed_ids)


async def _load_questions(room_id: str, question_ids: List[str]) -> Dict[str, ClusterQuestionItem]:
    if not question_ids:
        return {}

    redis = await get_redis()
    pipe = redis.pipeline()
    for qid in question_ids:
        pipe.hgetall(_question_key(room_id, qid))

    hashes = await pipe.execute()

    result: Dict[str, ClusterQuestionItem] = {}
    for h in hashes:
        if not h:
            continue
        try:
            question = ClusterQuestionItem(
                id=h["id"],
                content=h.get("content", ""),
                slide=int(h.get("slide", 1)),
                ts=int(h.get("ts", 0)),
                status=h.get("status", "active"),
            )
            result[question.id] = question
        except (KeyError, TypeError, ValueError):
            continue

    return result


async def _cluster_items(room_id: str, clusters: List[Dict]) -> List[ClusterItem]:
    inactive = await _inactive_ids(room_id)
    visible_ids: List[str] = []

    for c in clusters:
        member_ids = _dedupe(c.get("member_ids", []))
        visible_ids.extend(qid for qid in member_ids if qid not in inactive)

    questions_by_id = await _load_questions(room_id, _dedupe(visible_ids))
    items: List[ClusterItem] = []

    for c in sorted(clusters, key=lambda item: item.get("count", 0), reverse=True):
        member_ids = _dedupe(c.get("member_ids", []))
        questions = [
            questions_by_id[qid]
            for qid in member_ids
            if qid not in inactive
            and qid in questions_by_id
            and questions_by_id[qid].status == "active"
        ]
        if len(questions) < 2:
            continue

        representative = questions[0]
        question_ids = [q.id for q in questions]
        slides = sorted({q.slide for q in questions})
        samples = [q.content for q in questions[:3]]

        items.append(ClusterItem(
            clusterId=c.get("cluster_id") or c.get("representative_id") or question_ids[0],
            representativeQuestionId=representative.id,
            representative=representative.content,
            count=len(questions),
            questions=questions,
            questionIds=question_ids,
            slides=slides,
            samples=samples,
        ))

    return sorted(items, key=lambda item: item.count, reverse=True)


async def refresh_clusters(room_id: str) -> ClusterReportResponse:
    # 완료/삭제된 질문을 Redis 클러스터 상태에서 제거하고 현재 활성 클러스터 응답을 반환한다.
    redis = await get_redis()
    raw = await redis.get(_clusters_key(room_id))
    clusters: List[Dict] = json.loads(raw) if raw else []
    inactive = await _inactive_ids(room_id)
    active_ids = _dedupe([
        qid
        for c in clusters
        for qid in c.get("member_ids", [])
        if qid not in inactive
    ])
    questions_by_id = await _load_questions(room_id, active_ids)

    compacted = []
    for c in clusters:
        member_ids = [qid for qid in _dedupe(c.get("member_ids", [])) if qid not in inactive]
        questions = [
            questions_by_id[qid]
            for qid in member_ids
            if qid in questions_by_id and questions_by_id[qid].status == "active"
        ]
        if not questions:
            continue

        representative = questions[0]
        embeddings = [_embed(q.content) for q in questions]
        centroid = np.mean(embeddings, axis=0)
        norm = TS.normalize(representative.content)
        ngrams = TS.char_ngrams(norm, NGRAM)

        c["member_ids"] = [q.id for q in questions]
        c["representative"] = representative.content
        c["representative_id"] = representative.id
        c["cluster_id"] = c.get("cluster_id") or representative.id
        c["slides"] = sorted({q.slide for q in questions})
        c["samples"] = [q.content for q in questions[:3]]
        c["count"] = len(questions)
        c["centroid_emb"] = centroid.tolist()
        c["simhash"] = TS.simhash64(ngrams)
        c["ngrams"] = list(ngrams)
        compacted.append(c)

    await redis.set(_clusters_key(room_id), json.dumps(compacted))
    return await get_current_clusters(room_id)


async def get_current_clusters(room_id: str) -> ClusterReportResponse:
    # 새 질문 추가 없이 Redis에 저장된 현재 클러스터 상태를 그대로 읽어 반환한다.
    # 발표자 페이지 새로고침 시 초기 상태 복원에 사용된다.
    redis = await get_redis()

    raw = await redis.get(_clusters_key(room_id))
    clusters: List[Dict] = json.loads(raw) if raw else []

    total_raw = await redis.get(_total_key(room_id))
    total = int(total_raw) if total_raw else 0

    items = await _cluster_items(room_id, clusters)

    return ClusterReportResponse(
        roomId=room_id,
        totalQuestions=total,
        uniqueGroups=len(items),
        clusters=items,
    )


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
        training_representative: str | None = None

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
            # 코사인 유사도 기준 합류: 기존 중심과 새 임베딩을 가중 평균해 중심 안정화
            c = clusters[best_cos_idx]
            training_representative = c["representative"]
            c["member_ids"].append(question.id)
            c["slides"] = sorted(set(c["slides"] + [question.slide]))
            if len(c["samples"]) < 3:
                c["samples"].append(question.content)
            c["count"] += 1
            old_centroid = np.array(c["centroid_emb"], dtype=np.float32)
            c["centroid_emb"] = ((old_centroid * (c["count"] - 1) + emb) / c["count"]).tolist()
            joined = True

        elif best_d_idx >= 0 and best_d <= HAMMING_THRESHOLD:
            # 해밍 거리 기준 합류: 문자 패턴이 비슷한 경우
            c = clusters[best_d_idx]
            training_representative = c["representative"]
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
                    training_representative = c["representative"]
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
                    "representative_id": question.id,
                    "cluster_id": question.id,
                    "centroid_emb": emb.tolist(),
                    "simhash": simh,
                    "ngrams": list(sh),
                    "member_ids": [question.id],
                    "slides": [question.slide],
                    "samples": [question.content],
                    "count": 1,
                })

        # 합류 성공 시 파인튜닝용 데이터를 MySQL에 저장
        if joined and training_representative is not None:
            async with async_session_factory() as session:
                session.add(TrainingData(
                    room_id=room_id,
                    question_a=question.content,
                    question_b=training_representative,
                    is_similar=True,
                ))
                await session.commit()

        # 갱신된 클러스터 상태를 Redis에 저장
        await redis.set(_clusters_key(room_id), json.dumps(clusters))

        # 전체 질문 수는 Spring Boot가 관리하는 카운터에서 읽음
        total_raw = await redis.get(_total_key(room_id))
        total = int(total_raw) if total_raw else 0

        items = await _cluster_items(room_id, clusters)

        logger.info(f"[IncrementalCluster] roomId={room_id}, 총 {len(clusters)}개 그룹")

        return ClusterReportResponse(
            roomId=room_id,
            totalQuestions=total,
            uniqueGroups=len(items),
            clusters=items,
        )

    except AppException:
        raise
    except Exception as e:
        logger.exception(f"[IncrementalCluster] 알 수 없는 오류: {e}")
        raise AppException(ReportErrorCode.UNKNOWN, detail=str(e))
