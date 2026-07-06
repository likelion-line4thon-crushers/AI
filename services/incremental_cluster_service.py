import json
import logging
import numpy as np
from typing import List, Dict, Set

from sqlalchemy import select, and_, or_

from models.cluster import QuestionInput, ClusterItem, ClusterQuestionItem, ClusterReportResponse
from models.training_data import TrainingData
from services import text_sim as TS
from services.llm_judge import judge_same
from core.redis import get_redis
from core.db import async_session_factory
from exception.errors import AppException, ReportErrorCode
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMB_MODEL = "BAAI/bge-m3"  # 다국어(한국어 포함) 임베딩. KR-SBERT 대비 평가셋 F1 0.056→0.716
# 회색지대 판정 임계값 (실험으로 확정, bge-m3 기준):
#   cosine >= EMB_HIGH            → 자동 합류
#   EMB_LOW <= cosine < EMB_HIGH  → gpt-4o 판정 (같으면 합류, 아니면/실패면 신규)
#   cosine < EMB_LOW              → 자동 신규
EMB_HIGH = 0.62
EMB_LOW = 0.50

_model = SentenceTransformer(EMB_MODEL)
_EMB_DIM = _model.get_sentence_embedding_dimension()  # 현재 모델의 임베딩 차원 (bge-m3=1024)


def _embed(text: str) -> np.ndarray:
    # 정규화 → SBERT 임베딩 (코사인 정규화 포함).
    # 형태소 키워드 추출은 문장형 임베딩 모델 품질을 떨어뜨려 제거함(평가셋에서 확인).
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

        c["member_ids"] = [q.id for q in questions]
        c["representative"] = representative.content
        c["representative_id"] = representative.id
        c["cluster_id"] = c.get("cluster_id") or representative.id
        c["slides"] = sorted({q.slide for q in questions})
        c["samples"] = [q.content for q in questions[:3]]
        c["count"] = len(questions)
        c["centroid_emb"] = centroid.tolist()
        c.pop("simhash", None)   # 표면형 fallback 제거로 더 이상 저장하지 않음
        c.pop("ngrams", None)
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


async def _pair_exists(session, question_a: str, question_b: str) -> bool:
    # 같은 쌍이 이미 저장돼 있는지 확인 (중복 저장 방지).
    # 유사도는 대칭이므로 (A,B)와 (B,A)를 같은 쌍으로 보고 순서 무관하게 조회한다.
    # room_id는 조건에 넣지 않는다 — 학습/평가셋은 '질문 텍스트 쌍' 단위라 방 구분 없이 전역 dedup 한다.
    result = await session.execute(
        select(TrainingData.id)
        .where(or_(
            and_(TrainingData.question_a == question_a, TrainingData.question_b == question_b),
            and_(TrainingData.question_a == question_b, TrainingData.question_b == question_a),
        ))
        .limit(1)
    )
    return result.first() is not None


async def _save_training_pair(room_id: str, question_a: str, question_b: str,
                              is_similar: bool, cosine: float) -> None:
    # 회색지대 gpt-4o 판정 쌍을 학습/평가용으로 저장. 부수 기능이라 실패해도 요청을 깨지 않는다.
    try:
        async with async_session_factory() as session:
            if await _pair_exists(session, question_a, question_b):
                return
            session.add(TrainingData(
                room_id=room_id,
                question_a=question_a,
                question_b=question_b,
                is_similar=is_similar,
                cosine=cosine,
            ))
            await session.commit()
    except Exception as e:
        logger.warning(f"[학습데이터] 저장 실패(무시): {e!r}")


async def add_question_to_clusters(room_id: str, question: QuestionInput) -> ClusterReportResponse:
    # 새 질문 1개를 기존 클러스터 상태에 증분 추가하고 전체 클러스터 결과를 반환한다.
    # 판단: 최근접 클러스터와의 코사인이 EMB_HIGH 이상 자동 합류 / EMB_LOW~EMB_HIGH 회색지대는
    #       gpt-4o 판정(judge_same) / EMB_LOW 미만 자동 신규. (표면형 fallback은 오합류로 제거함)
    # 클러스터 상태는 Redis(room:{roomId}:clusters)에 JSON으로 저장/갱신된다.
    redis = await get_redis()

    try:
        # 빈/기호-only 질문은 임베딩하지 않고 바로 신규 클러스터로 처리한다.
        # (정규화 후 빈 문자열이면 무의미한 임베딩이 되어 오합류를 유발하므로)
        emb = _embed(question.content) if TS.normalize(question.content) else None

        # 기존 클러스터 상태 로드
        raw = await redis.get(_clusters_key(room_id))
        clusters: List[Dict] = json.loads(raw) if raw else []

        # 임베딩 모델 교체 등으로 기존 상태의 centroid 차원이 현재 모델과 다르면,
        # 과거 상태를 신뢰할 수 없으므로 전체 클러스터를 리셋하고 새로 시작한다.
        # (모델은 한 번에 교체되는 구조라 첫 클러스터만 확인하면 충분)
        if clusters and len(clusters[0].get("centroid_emb", [])) != _EMB_DIM:
            logger.warning(
                f"[IncrementalCluster] roomId={room_id}: centroid 차원 불일치 "
                f"({len(clusters[0].get('centroid_emb', []))} != {_EMB_DIM}) → 클러스터 상태 리셋"
            )
            clusters = []

        joined = False
        best_cos_idx = -1
        best_cos = -1.0
        judged_rep: str | None = None       # 회색지대에서 비교한 대표 질문
        judged_verdict: bool | None = None  # gpt-4o 판정 결과 (True/False, 실패면 None)

        # 전체 클러스터와 코사인 유사도 비교해서 가장 가까운 클러스터 탐색
        # (빈 질문 emb=None이면 비교를 건너뛰고 신규 클러스터로 생성)
        if emb is not None:
            for i, c in enumerate(clusters):
                cent_emb = np.array(c["centroid_emb"], dtype=np.float32)
                cos = float(emb @ cent_emb)
                if cos > best_cos:
                    best_cos = cos
                    best_cos_idx = i

        # 합류 여부 판정: HIGH 이상 자동 합류 / 회색지대는 gpt-4o 판정 / LOW 미만 자동 신규
        should_join = False
        if best_cos_idx >= 0:
            if best_cos >= EMB_HIGH:
                should_join = True
                logger.info(f"[군집화] roomId={room_id} cos={best_cos:.3f} >= {EMB_HIGH} → 자동 합류")
            elif best_cos >= EMB_LOW:
                # 회색지대: gpt-4o 판정. 실패/타임아웃/파싱실패(None) 또는 '다르다'(False) → 신규로 폴백.
                judged_rep = clusters[best_cos_idx]["representative"]
                judged_verdict = await judge_same(question.content, judged_rep)
                should_join = judged_verdict is True
                logger.info(
                    f"[군집화] roomId={room_id} cos={best_cos:.3f} 회색지대 판정={judged_verdict} "
                    f"→ {'합류' if should_join else '신규'}"
                )
            # best_cos < EMB_LOW 이면 should_join=False (자동 신규)

        if should_join:
            # 기존 중심과 새 임베딩을 가중 평균해 중심 안정화
            c = clusters[best_cos_idx]
            c["member_ids"].append(question.id)
            c["slides"] = sorted(set(c["slides"] + [question.slide]))
            if len(c["samples"]) < 3:
                c["samples"].append(question.content)
            c["count"] += 1
            old_centroid = np.array(c["centroid_emb"], dtype=np.float32)
            # 가중 평균. 단위벡터 평균이라 norm<1이 되지만 재정규화는 의도적으로 보류한다:
            # 재정규화하면 임계값(EMB_HIGH/EMB_LOW) 의미가 바뀌어 재튜닝이 필요하고,
            # 현재 평가셋 F1도 이 동작 기준이라 임계값 재튜닝과 묶어 별도로 처리한다.
            c["centroid_emb"] = ((old_centroid * (c["count"] - 1) + emb) / c["count"]).tolist()
            joined = True

        if not joined:
            # 어느 클러스터에도 속하지 않으면 신규 클러스터 생성.
            # 빈 질문(emb=None)은 영벡터를 중심으로 둬 이후 코사인이 항상 0 → 다른 질문과 섞이지 않음.
            centroid = emb if emb is not None else np.zeros(_EMB_DIM, dtype=np.float32)
            clusters.append({
                "representative": question.content,
                "representative_id": question.id,
                "cluster_id": question.id,
                "centroid_emb": centroid.tolist(),
                "member_ids": [question.id],
                "slides": [question.slide],
                "samples": [question.content],
                "count": 1,
            })

        # 회색지대 gpt-4o 판정 쌍만 학습/평가용으로 저장한다 (같다=positive / 다르다=negative).
        # 자동 합류(>=HIGH)·자동 신규(<LOW)는 gpt-4o 판정이 없고, 폴백(judged_verdict=None)은
        # 라벨이 없으므로 저장하지 않는다.
        if judged_verdict is not None and judged_rep is not None:
            await _save_training_pair(
                room_id, question.content, judged_rep, judged_verdict, best_cos,
            )

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
