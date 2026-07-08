"""
실제 서비스 코드 경로(incremental_cluster_service.add_question_to_clusters)로 평가.

측정 스크립트(run_embed_compare.py)의 재현 클러스터링이 아니라, 운영에서 실제로 도는
add_question_to_clusters 를 그대로 호출한다:
  - 임베딩: 서비스의 _embed (진짜 KR-SBERT, TS.normalize + normalize_embeddings=True)
  - 밴드:   서비스의 EMB_HIGH / EMB_LOW (현재 0.50 / 0.35)
  - 회색지대 판정: 서비스의 judge_same (진짜 gpt-4o, llm_judge.py 프롬프트)
  - centroid/차원가드/라우팅: 전부 서비스 로직 그대로
Redis / DB 만 인메모리 fake 로 대체(네트워크 제거)하고, gold 질문을 순서대로 투입한 뒤
최종 Redis 클러스터 상태(member_ids)로 pairwise F1 을 계산한다.

실행: .venv/Scripts/python.exe eval/run_service_eval.py
"""
import asyncio
import contextlib
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import run_eval as E  # load_gold / pairwise_metrics 재사용  # noqa: E402
from models.cluster import QuestionInput  # noqa: E402


# ── 인메모리 fake Redis (서비스가 쓰는 최소 API만) ──────────────
class _FakePipe:
    def __init__(self, redis):
        self._redis = redis
        self._q = []

    def hgetall(self, key):
        self._q.append(key)
        return self

    async def execute(self):
        return [dict(self._redis._hashes.get(k, {})) for k in self._q]


class FakeRedis:
    def __init__(self):
        self._kv = {}
        self._sets = {}
        self._hashes = {}

    async def get(self, key):
        return self._kv.get(key)

    async def set(self, key, value):
        self._kv[key] = value

    async def smembers(self, key):
        return set(self._sets.get(key, set()))

    def pipeline(self):
        return _FakePipe(self)


# ── 인메모리 fake DB 세션 (training_data 저장 경로 무해화) ──────
class _FakeSession:
    async def execute(self, statement):
        class _R:
            def first(self_inner):
                return None
        return _R()

    def add(self, obj):
        pass

    async def commit(self):
        pass


def _fake_session_factory():
    @contextlib.asynccontextmanager
    async def _cm():
        yield _FakeSession()
    return _cm()


async def main():
    import services.incremental_cluster_service as mod

    print(f"[service] EMB_MODEL={mod.EMB_MODEL}")
    print(f"[service] EMB_HIGH={mod.EMB_HIGH}  EMB_LOW={mod.EMB_LOW}  _EMB_DIM={mod._EMB_DIM}")

    # Redis/DB 만 fake 로 교체. _embed(KR-SBERT)·judge_same(gpt-4o) 는 실제 그대로 둔다.
    fake_redis = FakeRedis()

    async def _get_redis():
        return fake_redis

    mod.get_redis = _get_redis
    mod.async_session_factory = _fake_session_factory

    # 회색지대 gpt-4o 호출 횟수 카운트 (실제 judge 를 감싸기만 함)
    real_judge = mod.judge_same
    calls = [0]

    async def _counting_judge(a, b):
        calls[0] += 1
        return await real_judge(a, b)

    mod.judge_same = _counting_judge

    gold_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_questions.jsonl")
    gold = E.load_gold(gold_path)
    print(f"[load] 질문 {len(gold)}개, 정답 group {len(set(q['group'] for q in gold))}개")

    room = "svc-eval"
    for q in gold:
        await mod.add_question_to_clusters(
            room,
            QuestionInput(id=q["id"], content=q["content"], slide=q.get("slide", 1), ts=q.get("ts", 0)),
        )

    # 최종 서비스 클러스터 상태 → pairwise F1 (싱글턴 포함 전체 클러스터 사용)
    raw = fake_redis._kv.get(f"room:{room}:clusters")
    clusters = json.loads(raw) if raw else []
    id_to_idx = {q["id"]: i for i, q in enumerate(gold)}
    pred = [[id_to_idx[mid] for mid in c["member_ids"] if mid in id_to_idx] for c in clusters]
    gold_groups = [q["group"] for q in gold]
    m = E.pairwise_metrics(pred, gold_groups)

    print("\n=== 실제 서비스 코드 경로 평가 (KR-SBERT + 밴드 0.50/0.35 + gpt-4o) ===")
    print(f"  예측 클러스터 수: {len(pred)}  (정답 group 수: {len(set(gold_groups))})")
    print(f"  precision={m['precision']:.3f}  recall={m['recall']:.3f}  F1={m['f1']:.3f}"
          f"   (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
    print(f"  회색지대 gpt-4o 호출: {calls[0]}회 / 전체 {len(gold)}개 질문")
    print(f"\n  [비교] run_embed_compare.py KR-SBERT 회색지대 best F1 = 0.600 (밴드 0.35/0.50)")
    delta = m["f1"] - 0.600
    print(f"  [차이] {m['f1']:.3f} - 0.600 = {delta:+.3f}")


if __name__ == "__main__":
    asyncio.run(main())
