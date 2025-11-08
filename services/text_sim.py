import re
from collections import Counter

import unicodedata
from typing import Iterable, Set, Tuple, List, Dict
import math


_SPACE_MULTI = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

def normalize(s: str | None) -> str:
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", s).lower().replace("\u00A0", " ")
    t = _PUNCT.sub(" ", t)
    t = _SPACE_MULTI.sub(" ", t).strip()
    return t

def char_ngrams(s: str, n: int = 2) -> Set[str]:
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s)-n+1)}

def murmur64(data: str) -> int:
    b = data.encode("utf-8")
    h = 0xC70F6907
    for x in b:
        h ^= x
        h = (h * 0x5BD1E9955BD1E995) & 0xFFFFFFFFFFFFFFFF
        h ^= (h >> 47)
    return h & 0xFFFFFFFFFFFFFFFF

def simhash64(features: Iterable[str]) -> int:
    v = [0] * 64
    anyf = False
    for f in features:
        anyf = True
        h = murmur64(f)
        for i in range(64):
            v[i] += 1 if ((h >> i) & 1) else -1
    if not anyf:
        return 0
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def char_ngrams_multi(s: str, n_range: Tuple[int, int] = (2, 4)) -> List[str]:
    # 공백 제거 후 2~4gram 문자 n-gram 토큰 리스트 생성
    t = normalize(s).replace(" ", "")
    if not t:
        return []
    out: List[str] = []
    n_min, n_max = n_range
    L = len(t)
    for n in range(n_min, n_max + 1):
        if L >= n:
            out.extend(t[i:i+n] for i in range(L - n + 1))
    return out

def build_idf(corpus_tokens: List[List[str]]) -> Dict[str, float]:
    # 코퍼스(질문 리스트) 전체에 대한 DF → IDF 맵 (불용어 없이도 흔한 조각이 자동 감쇠)
    # idf = log((N + 1) / (df + 1)) + 1  (스무딩)
    N = len(corpus_tokens)
    df: Counter = Counter()
    for toks in corpus_tokens:
        df.update(set(toks))
    idf = {t: math.log((N + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}
    return idf

def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    # 토큰의 TF * IDF 벡터(사전)
    tf = Counter(tokens)
    return {t: tf[t] * idf.get(t, 1.0) for t in tf}

def cosine_dict(a: Dict[str, float], b: Dict[str, float]) -> float:
    # 사전 벡터 코사인 유사도 (sparse)
    if not a or not b:
        return 0.0
    # dot
    (small, big) = (a, b) if len(a) < len(b) else (b, a)
    dot = sum(w * big.get(t, 0.0) for t, w in small.items())
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return (dot / (na * nb)) if (na and nb) else 0.0