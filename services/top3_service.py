# services/top3_service.py
from typing import List, Set, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from models.question_report import QuestionRecord, TopQuestionItem, TopQuestionReportResponse
from . import text_sim as TS

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
    return _model.encode(TS.normalize(text), normalize_embeddings=True)

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    # 코사인 유사도 계산 (정규화되어 있으면 내적이 코사인)
    return float(a @ b)

# 내부 클래스
class _Q:
    __slots__ = ("q", "norm", "sh", "simh", "emb")
    def __init__(self, q: QuestionRecord):
        self.q = q
        self.norm = TS.normalize(q.content)
        self.sh: Set[str] = TS.char_ngrams(self.norm, NGRAM)
        self.simh = TS.simhash64(self.sh)
        self.emb = _embed(q.content)

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
def build_top3(questions: List[QuestionRecord]) -> TopQuestionReportResponse:
    if not questions:
        return TopQuestionReportResponse(totalQuestions=0, uniqueGroups=0, top3=[])

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
                # --- 의미 유사도 기준으로 합류 ---
                c = best
                c.members.append(cur)
                c.slides.add(cur.q.slide)
                c.ids.add(cur.q.id)
                c.samples.append(cur.q.content)
                c.cent_emb = cur.emb
                joined = True

            elif best and best_d <= HAMMING_THRESHOLD:
                c = best
                c.members.append(cur)
                c.slides.add(cur.q.slide)
                c.ids.add(cur.q.id)
                c.samples.append(cur.q.content)
                joined = True

            else:
                # --- 자카드 fallback ---
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
        reverse=True
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

    return TopQuestionReportResponse(
        totalQuestions=len(questions),
        uniqueGroups=len(clusters),
        top3=top3,
    )
