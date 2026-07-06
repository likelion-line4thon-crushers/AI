"""
add_question_to_clusters 동작 테스트 (이번 리뷰에서 고친 부분 위주).
가짜 임베딩(conftest)으로 코사인을 제어하고, fake redis 에 저장된 상태를 검증한다.
"""
import json

from models.cluster import QuestionInput


def _q(qid, content, slide=1, ts=0):
    return QuestionInput(id=qid, content=content, slide=slide, ts=ts)


def _stored_clusters(fake_redis, room):
    raw = fake_redis._kv.get(f"room:{room}:clusters")
    return json.loads(raw) if raw else []


# ── 1) 차원 불일치 리셋 ──────────────────────────────────────
async def test_dimension_mismatch_resets_stale_clusters(svc):
    """옛 차원(3) centroid가 저장돼 있으면 새 질문(차원4) 처리 시 전체 리셋된다."""
    room = "r1"
    old = {
        "representative": "old", "representative_id": "old", "cluster_id": "old",
        "centroid_emb": [0.1, 0.2, 0.3],  # 옛 차원 3 (bge-m3 이전 상태 모사)
        "member_ids": ["old"], "slides": [1], "samples": ["old"], "count": 1,
    }
    svc.redis._kv[f"room:{room}:clusters"] = json.dumps([old])

    await svc.mod.add_question_to_clusters(room, _q("q1", "cat one"))

    clusters = _stored_clusters(svc.redis, room)
    assert len(clusters) == 1                                     # 옛 클러스터 제거됨
    assert clusters[0]["member_ids"] == ["q1"]                    # 옛 것에 합류하지 않고 신규
    assert len(clusters[0]["centroid_emb"]) == svc.mod._EMB_DIM   # 새 차원(4)


async def test_cluster_missing_centroid_key_is_safe(svc):
    """centroid_emb 키가 없는 malformed 상태에서도 KeyError 없이 리셋된다."""
    room = "r2"
    svc.redis._kv[f"room:{room}:clusters"] = json.dumps([{"member_ids": ["x"]}])  # centroid_emb 없음

    await svc.mod.add_question_to_clusters(room, _q("q1", "cat one"))

    clusters = _stored_clusters(svc.redis, room)
    assert len(clusters) == 1
    assert clusters[0]["member_ids"] == ["q1"]


async def test_empty_redis_state_creates_first_cluster(svc):
    """Redis 키 자체가 없을 때(get→None)도 크래시 없이 첫 클러스터를 만든다."""
    room = "r3"  # 아무것도 seed 안 함

    await svc.mod.add_question_to_clusters(room, _q("q1", "cat one"))

    clusters = _stored_clusters(svc.redis, room)
    assert len(clusters) == 1
    assert clusters[0]["member_ids"] == ["q1"]


# ── 2) 빈/기호-only 입력 ─────────────────────────────────────
async def test_empty_content_skips_embedding_and_is_isolated(svc):
    """정규화 후 빈 문자열이면 임베딩을 건너뛰고 영벡터 신규 클러스터로 격리한다.

    가짜 모델은 빈 문자열("")에 대해 기존 클러스터(cat one)와 동일한 벡터를 준다.
    따라서 빈 입력 가드가 없다면 encode 가 호출되어 기존 클러스터에 '잘못 합류'하게 되고
    (clusters 1개, count 2), 가드가 있으면 encode 자체가 생략되고 신규로 격리된다(clusters 2개).
    """
    room = "r4"
    existing = {
        "representative": "cat one", "representative_id": "e1", "cluster_id": "e1",
        "centroid_emb": [1.0, 0.0, 0.0, 0.0],  # 차원 4 → 리셋되지 않음
        "member_ids": ["e1"], "slides": [1], "samples": ["cat one"], "count": 1,
    }
    svc.redis._kv[f"room:{room}:clusters"] = json.dumps([existing])

    await svc.mod.add_question_to_clusters(room, _q("q_empty", "!!!"))  # normalize -> ""

    clusters = _stored_clusters(svc.redis, room)
    assert len(clusters) == 2                                   # 기존에 합류하지 않고 신규 생성
    existing_after = next(c for c in clusters if c["member_ids"] == ["e1"])
    empty_after = next(c for c in clusters if c["member_ids"] == ["q_empty"])
    assert existing_after["count"] == 1                         # 기존 클러스터를 건드리지 않음
    assert empty_after["centroid_emb"] == [0.0] * svc.mod._EMB_DIM  # 영벡터 중심
    # 가드가 실제로 임베딩을 우회했는지 검증 (mock 우회로 항상 통과하는 것을 방지)
    assert "" not in svc.mod._model.encoded


async def test_single_member_cluster_not_exposed_in_response(svc):
    """멤버 1개짜리 클러스터는 응답(clusters)에 노출되지 않는다."""
    room = "r7"
    svc.redis.seed_hash(
        f"room:{room}:question:q1",
        {"id": "q1", "content": "cat one", "slide": "1", "ts": "0", "status": "active"},
    )

    resp = await svc.mod.add_question_to_clusters(room, _q("q1", "cat one"))

    assert resp.uniqueGroups == 0
    assert resp.clusters == []


# ── 3) 정상 합류 (sanity) ────────────────────────────────────
async def test_similar_questions_join_same_cluster(svc):
    """임계값을 넘는 유사 입력 2개(cat one/cat two)가 같은 클러스터로 묶인다."""
    room = "r5"

    await svc.mod.add_question_to_clusters(room, _q("q1", "cat one"))
    await svc.mod.add_question_to_clusters(room, _q("q2", "cat two"))

    clusters = _stored_clusters(svc.redis, room)
    assert len(clusters) == 1
    assert clusters[0]["member_ids"] == ["q1", "q2"]
    assert clusters[0]["count"] == 2


async def test_dissimilar_question_forms_new_cluster(svc):
    """임계값 미만 입력(dog)은 별도 클러스터가 된다."""
    room = "r6"

    await svc.mod.add_question_to_clusters(room, _q("q1", "cat one"))
    await svc.mod.add_question_to_clusters(room, _q("q2", "dog"))

    clusters = _stored_clusters(svc.redis, room)
    assert len(clusters) == 2


async def test_two_member_cluster_is_exposed_in_response(svc):
    """멤버 2개 이상 클러스터는 응답에 노출되고 count가 맞다."""
    room = "r8"
    for qid, content in [("q1", "cat one"), ("q2", "cat two")]:
        svc.redis.seed_hash(
            f"room:{room}:question:{qid}",
            {"id": qid, "content": content, "slide": "1", "ts": "0", "status": "active"},
        )

    await svc.mod.add_question_to_clusters(room, _q("q1", "cat one"))
    resp = await svc.mod.add_question_to_clusters(room, _q("q2", "cat two"))

    assert resp.uniqueGroups == 1
    assert len(resp.clusters) == 1
    assert resp.clusters[0].count == 2
