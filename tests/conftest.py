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
        "cat two": [0.9, 0.1, 0.0, 0.0],   # cat one 과 코사인 ~0.99 → 합류
        "dog": [0.0, 1.0, 0.0, 0.0],       # cat one 과 코사인 0 → 신규
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


# ── 3) Fake DB 세션 (합류 시 TrainingData 저장 경로) ──────────
class _FakeSession:
    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass


def make_fake_session_factory(recorder):
    @contextlib.asynccontextmanager
    async def _cm():
        sess = _FakeSession()
        try:
            yield sess
        finally:
            recorder.extend(sess.added)

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

    # 모듈 레벨 _model 은 세션 내내 공유되므로 테스트 간 encode 기록을 격리한다.
    mod._model.encoded.clear()

    return types.SimpleNamespace(mod=mod, redis=fake_redis, training_saved=training_saved)
