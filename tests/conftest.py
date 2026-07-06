"""
pytest 공통 픽스처 / mock.

핵심: services.incremental_cluster_service 는 import 시점에 bge-m3 를 로드하고
_EMB_DIM 을 읽는다. 그래서 서비스가 import 되기 "전에" sentence_transformers 를
가짜 모듈로 선점(sys.modules)해 모델 다운로드/로드 없이 빠르게 테스트한다.

가짜 임베딩은 차원 4의 고정 벡터로, 텍스트에 따라 코사인 유사도를 제어할 수 있다.
Redis / DB 도 네트워크 없이 인메모리 fake 로 대체한다.
"""
import contextlib
import os
import sys
import types

import numpy as np

# ── 0) repo 루트를 import 경로에 추가 ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── 1) sentence_transformers 선점 (서비스 import 전에!) ────────
class FakeSentenceTransformer:
    """텍스트 → 고정 벡터. normalize_embeddings=True 면 단위벡터로 정규화."""
    DIM = 4
    # 정규화(TS.normalize) 후의 텍스트를 키로 사용한다.
    VECTORS = {
        "cat one": [1.0, 0.0, 0.0, 0.0],
        "cat two": [0.9, 0.1, 0.0, 0.0],   # cat one 과 코사인 ~0.99 (>=0.62) → 자동 합류
        "dog": [0.0, 1.0, 0.0, 0.0],       # cat one 과 코사인 0 (<0.50) → 자동 신규
        "borderline": [0.55, 0.835, 0.0, 0.0],  # cat one 과 코사인 ~0.55 → 회색지대(LLM 판정)
        # 빈 문자열도 일부러 non-zero(= cat one 과 동일) 로 둔다. 실제 bge-m3 도 빈 입력에
        # 무의미한 non-zero 벡터를 주므로, 빈 입력 가드가 없으면 기존 클러스터에 잘못 합류한다.
        # 가드가 있으면 encode 자체가 호출되지 않아 이 벡터는 쓰이지 않는다.
        "": [1.0, 0.0, 0.0, 0.0],
    }

    def __init__(self, *args, **kwargs):
        self.encoded = []  # encode 로 들어온 텍스트 기록 (호출 스파이용)

    def get_sentence_embedding_dimension(self):
        return self.DIM

    def encode(self, text, normalize_embeddings=False, **kwargs):
        self.encoded.append(text)
        vec = np.asarray(self.VECTORS.get(text, [0.0] * self.DIM), dtype=np.float32)
        if normalize_embeddings:
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
        return vec


_fake_st = types.ModuleType("sentence_transformers")
_fake_st.SentenceTransformer = FakeSentenceTransformer
sys.modules["sentence_transformers"] = _fake_st


# ── 2) 인메모리 Fake Redis ────────────────────────────────────
class _FakePipeline:
    def __init__(self, redis):
        self._redis = redis
        self._queue = []

    def hgetall(self, key):
        self._queue.append(key)
        return self

    async def execute(self):
        return [dict(self._redis._hashes.get(k, {})) for k in self._queue]


class FakeAsyncRedis:
    def __init__(self):
        self._kv = {}       # 문자열 값
        self._sets = {}     # set 값
        self._hashes = {}   # hash 값 (질문 레코드)

    async def get(self, key):
        return self._kv.get(key)

    async def set(self, key, value):
        self._kv[key] = value

    async def smembers(self, key):
        return set(self._sets.get(key, set()))

    def pipeline(self):
        return _FakePipeline(self)

    # ── 테스트 셋업용 헬퍼 ──
    def seed_hash(self, key, mapping):
        self._hashes[key] = dict(mapping)


# ── 3) Fake DB 세션 (TrainingData 저장/중복확인 경로) ─────────
class _FakeResult:
    def __init__(self, matched):
        self._matched = matched

    def first(self):
        return (1,) if self._matched else None


class _FakeSession:
    """커밋된 행을 store(공유 리스트)에 쌓고, dedup select 를 흉내낸다."""
    def __init__(self, store):
        self._store = store
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self._store.extend(self.added)
        self.added = []

    async def execute(self, statement):
        # _pair_exists 의 WHERE 를 '위치 기반'으로 충실히 흉내낸다.
        # 바인드 파라미터를 (question_a_N, question_b_N) 브랜치로 복원해
        # (row.question_a==a and row.question_b==b) 를 브랜치별로 평가한다.
        # 이렇게 해야 순서민감/순서무관 쿼리를 실제와 동일하게 구분한다(집합 멤버십으로 뭉개지 않음).
        try:
            params = statement.compile().params
        except Exception:
            params = {}
        branches = {}  # suffix -> {"a": .., "b": ..}
        for key, val in params.items():
            if key.startswith("question_a"):
                branches.setdefault(key[len("question_a"):], {})["a"] = val
            elif key.startswith("question_b"):
                branches.setdefault(key[len("question_b"):], {})["b"] = val

        def row_matches(r):
            ra = getattr(r, "question_a", None)
            rb = getattr(r, "question_b", None)
            return any(
                "a" in br and "b" in br and ra == br["a"] and rb == br["b"]
                for br in branches.values()
            )

        matched = any(row_matches(r) for r in self._store)
        return _FakeResult(matched)


def make_fake_session_factory(store):
    @contextlib.asynccontextmanager
    async def _cm():
        yield _FakeSession(store)

    def _factory():
        return _cm()

    return _factory


# ── 4) 픽스처 ────────────────────────────────────────────────
import pytest  # noqa: E402


@pytest.fixture
def svc(monkeypatch):
    """
    incremental_cluster_service 모듈을 fake redis/DB 로 감싸 반환.
    반환: mod(모듈), redis(FakeAsyncRedis), training_saved(list) 를 담은 네임스페이스.
    """
    import services.incremental_cluster_service as mod

    fake_redis = FakeAsyncRedis()

    async def _get_redis():
        return fake_redis

    training_saved = []
    monkeypatch.setattr(mod, "get_redis", _get_redis)
    monkeypatch.setattr(mod, "async_session_factory", make_fake_session_factory(training_saved))

    # gpt-4o 판정은 실제 호출 없이 mock. judge_state["verdict"] 로 반환값을 제어하고,
    # judge_calls 로 호출 여부/인자를 검증한다. 기본 verdict=None(폴백=신규)이라
    # 회색지대를 타지 않는 테스트에서 실수로 실제 API가 불릴 일이 없다.
    judge_calls = []
    judge_state = {"verdict": None}

    async def _fake_judge(question_a, question_b):
        judge_calls.append((question_a, question_b))
        return judge_state["verdict"]

    monkeypatch.setattr(mod, "judge_same", _fake_judge)

    # 모듈 레벨 _model 은 세션 내내 공유되므로 테스트 간 encode 기록을 격리한다.
    mod._model.encoded.clear()

    return types.SimpleNamespace(
        mod=mod, redis=fake_redis, training_saved=training_saved,
        judge_calls=judge_calls, judge_state=judge_state,
    )
