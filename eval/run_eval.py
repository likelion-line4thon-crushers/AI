"""
합성 평가셋 기반 군집화 정확도 측정 하니스 (Phase 0).

- eval/gold_questions.jsonl 을 순서대로(증분 방식) 넣어 클러스터를 만든다.
- 예측 클러스터 vs 정답 group 을 pairwise precision/recall/F1 로 비교한다.
- 여러 전략(strategy)을 같은 평가셋 위에서 돌려 비교한다.

실행: 프로젝트 루트에서
    .venv/Scripts/python.exe eval/run_eval.py
"""
import json
import os
import sys
from itertools import combinations
from typing import Callable, Dict, List

import numpy as np

# Windows 콘솔 cp949 에서도 한국어/기호 출력이 깨지지 않도록
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 프로젝트 루트 import 경로
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer  # noqa: E402
from services import text_sim as TS  # noqa: E402

# ── 현재 운영값과 동일한 상수 ──────────────────────────────
EMB_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
EMB_THRESHOLD = 0.60
NGRAM = 2
HAMMING_THRESHOLD = 4
JACCARD_FALLBACK = 0.60

_MODELS: Dict[str, SentenceTransformer] = {}


def get_model(name: str) -> SentenceTransformer:
    if name not in _MODELS:
        print(f"[load] 임베딩 모델 로딩: {name} ...")
        _MODELS[name] = SentenceTransformer(name)
    return _MODELS[name]


def embed_keyword(text: str) -> np.ndarray:
    # 현재 incremental_cluster_service._embed 와 동일 (형태소 키워드 추출 후 임베딩)
    return get_model(EMB_MODEL).encode(TS.extract_keywords(TS.normalize(text)), normalize_embeddings=True)


def embed_raw(text: str) -> np.ndarray:
    # top3_service 방식 (정규화 원문 임베딩)
    return get_model(EMB_MODEL).encode(TS.normalize(text), normalize_embeddings=True)


def embed_bge(text: str) -> np.ndarray:
    # bge-m3 원문 임베딩 (dense, 코사인 정규화)
    return get_model("BAAI/bge-m3").encode(TS.normalize(text), normalize_embeddings=True)


def load_gold(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── 클러스터링 전략들 (모두 증분/순차 처리) ────────────────
def cluster_current(qs: List[Dict], embed: Callable[[str], np.ndarray], thr: float) -> List[List[int]]:
    """현재 add_question_to_clusters 로직 재현: 코사인 → 해밍 → 자카드 → 신규."""
    clusters: List[Dict] = []
    for idx, q in enumerate(qs):
        emb = embed(q["content"])
        norm = TS.normalize(q["content"])
        sh = TS.char_ngrams(norm, NGRAM)
        simh = TS.simhash64(sh)

        best_cos, best_cos_i = -1.0, -1
        best_d, best_d_i = 1 << 30, -1
        for i, c in enumerate(clusters):
            cos = float(emb @ c["centroid"])
            if cos > best_cos:
                best_cos, best_cos_i = cos, i
            d = TS.hamming(simh, c["simhash"])
            if d < best_d:
                best_d, best_d_i = d, i

        if best_cos_i >= 0 and best_cos >= thr:
            c = clusters[best_cos_i]
            c["members"].append(idx)
            n = len(c["members"])
            c["centroid"] = (c["centroid"] * (n - 1) + emb) / n
        elif best_d_i >= 0 and best_d <= HAMMING_THRESHOLD:
            clusters[best_d_i]["members"].append(idx)
        else:
            joined = False
            for c in clusters:
                if TS.jaccard(sh, c["ngrams"]) >= JACCARD_FALLBACK:
                    c["members"].append(idx)
                    joined = True
                    break
            if not joined:
                clusters.append({"centroid": emb, "simhash": simh, "ngrams": sh, "members": [idx]})
    return [c["members"] for c in clusters]


# ── 회색지대 LLM 판정 ─────────────────────────────────────
_llm_cache: Dict[tuple, bool] = {}
_llm_calls = [0]
_openai_client = None
_openai_model = None


def llm_same(a: str, b: str) -> bool:
    """두 질문이 본질적으로 같은 것을 묻는지 OpenAI로 판정 (캐시)."""
    global _openai_client, _openai_model
    key = tuple(sorted([a, b]))
    if key in _llm_cache:
        return _llm_cache[key]
    if _openai_client is None:
        from openai import OpenAI
        from config.settings import settings
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        _openai_model = settings.OPENAI_MODEL
    _llm_calls[0] += 1
    resp = _openai_client.chat.completions.create(
        model=_openai_model,
        messages=[
            {"role": "system", "content":
                "너는 발표 청중 질문을 군집화하는 판정기야. 두 질문이 '본질적으로 같은 것을 묻는' "
                "질문이면 same=true, 주제나 의도가 다르면 same=false. 반드시 JSON만 출력해."},
            {"role": "user", "content":
                f'질문1: "{a}"\n질문2: "{b}"\n두 질문이 같은 것을 묻고 있나요? '
                '{"same": true} 또는 {"same": false} 형식 JSON으로만 답하세요.'},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        val = bool(json.loads(resp.choices[0].message.content).get("same", False))
    except Exception:
        val = False
    _llm_cache[key] = val
    return val


def cluster_gray_zone(qs: List[Dict], embed: Callable[[str], np.ndarray],
                      low: float, high: float,
                      judge: Callable[[str, str], bool] = None) -> List[List[int]]:
    """코사인 >= high 자동합류 / < low 자동신규 / 그 사이(회색지대)만 judge 판정."""
    judge = judge or llm_same
    clusters: List[Dict] = []
    for idx, q in enumerate(qs):
        emb = embed(q["content"])
        best_cos, best_i = -1.0, -1
        for i, c in enumerate(clusters):
            cos = float(emb @ c["centroid"])
            if cos > best_cos:
                best_cos, best_i = cos, i

        join = False
        if best_i >= 0:
            if best_cos >= high:
                join = True
            elif best_cos >= low:
                join = judge(q["content"], clusters[best_i]["rep"])

        if join:
            c = clusters[best_i]
            c["members"].append(idx)
            n = len(c["members"])
            c["centroid"] = (c["centroid"] * (n - 1) + emb) / n
        else:
            clusters.append({"centroid": emb, "rep": q["content"], "members": [idx]})
    return [c["members"] for c in clusters]


def cluster_cosine_only(qs: List[Dict], embed: Callable[[str], np.ndarray], thr: float) -> List[List[int]]:
    """Phase 1 미리보기: 표면형 fallback 제거, 코사인만으로 판정."""
    clusters: List[Dict] = []
    for idx, q in enumerate(qs):
        emb = embed(q["content"])
        best_cos, best_i = -1.0, -1
        for i, c in enumerate(clusters):
            cos = float(emb @ c["centroid"])
            if cos > best_cos:
                best_cos, best_i = cos, i
        if best_i >= 0 and best_cos >= thr:
            c = clusters[best_i]
            c["members"].append(idx)
            n = len(c["members"])
            c["centroid"] = (c["centroid"] * (n - 1) + emb) / n
        else:
            clusters.append({"centroid": emb, "members": [idx]})
    return [c["members"] for c in clusters]


# ── 평가 지표 (pairwise) ──────────────────────────────────
def pairwise_metrics(pred_clusters: List[List[int]], gold_groups: List[str]) -> Dict[str, float]:
    n = len(gold_groups)
    pred_of = [0] * n
    for cid, members in enumerate(pred_clusters):
        for m in members:
            pred_of[m] = cid

    tp = fp = fn = tn = 0
    for i, j in combinations(range(n), 2):
        same_pred = pred_of[i] == pred_of[j]
        same_gold = gold_groups[i] == gold_groups[j]
        if same_pred and same_gold:
            tp += 1
        elif same_pred and not same_gold:
            fp += 1
        elif not same_pred and same_gold:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def false_merges(pred_clusters: List[List[int]], qs: List[Dict]) -> List[str]:
    """한 클러스터 안에 서로 다른 정답 group 이 섞인 경우(오합류) 리포트."""
    out = []
    for members in pred_clusters:
        groups = {qs[m]["group"] for m in members}
        if len(groups) > 1:
            desc = ", ".join(f'{qs[m]["id"]}({qs[m]["group"]})' for m in members)
            out.append(desc)
    return out


def run(name: str, pred: List[List[int]], qs: List[Dict]):
    gold_groups = [q["group"] for q in qs]
    n_gold = len(set(gold_groups))
    m = pairwise_metrics(pred, gold_groups)
    print(f"\n=== {name} ===")
    print(f"  예측 클러스터 수: {len(pred)}  (정답 group 수: {n_gold})")
    print(f"  precision={m['precision']:.3f}  recall={m['recall']:.3f}  F1={m['f1']:.3f}"
          f"   (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
    fm = false_merges(pred, qs)
    if fm:
        print(f"  [!] 오합류(다른 그룹이 한 클러스터에 섞임) {len(fm)}건:")
        for d in fm:
            print(f"      - {d}")
    return m


def main():
    gold_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_questions.jsonl")
    qs = load_gold(gold_path)
    print(f"[load] 질문 {len(qs)}개, 정답 group {len(set(q['group'] for q in qs))}개")

    run("BASELINE (현재: 키워드임베딩 + 해밍/자카드 fallback, thr=0.60)",
        cluster_current(qs, embed_keyword, EMB_THRESHOLD), qs)

    run("A. fallback 제거 (키워드임베딩, 코사인만, thr=0.60)",
        cluster_cosine_only(qs, embed_keyword, EMB_THRESHOLD), qs)

    run("B. fallback 제거 + 원문임베딩 (코사인만, thr=0.60)",
        cluster_cosine_only(qs, embed_raw, EMB_THRESHOLD), qs)

    # KR-SBERT 원문임베딩 임계값 스윕
    print("\n[임계값 스윕 · KR-SBERT 원문임베딩 · 코사인만]")
    for thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        pred = cluster_cosine_only(qs, embed_raw, thr)
        m = pairwise_metrics(pred, [q["group"] for q in qs])
        print(f"  thr={thr:.2f}  clusters={len(pred):2d}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"(FP={m['fp']} FN={m['fn']})")

    # bge-m3 임계값 스윕 (다운로드 완료 시)
    try:
        print("\n[임계값 스윕 · bge-m3 원문임베딩 · 코사인만]")
        best = None
        for thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            pred = cluster_cosine_only(qs, embed_bge, thr)
            m = pairwise_metrics(pred, [q["group"] for q in qs])
            print(f"  thr={thr:.2f}  clusters={len(pred):2d}  "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                  f"(FP={m['fp']} FN={m['fn']})")
            if best is None or m["f1"] > best[1]["f1"]:
                best = (thr, m, pred)
        if best:
            run(f"bge-m3 BEST 코사인만 (thr={best[0]:.2f})", best[2], qs)
    except Exception as e:
        print(f"  [skip] bge-m3 측정 불가: {e}")

    # bge-m3 + 회색지대 LLM 판정 (low/high 밴드 몇 개 비교)
    try:
        print("\n[bge-m3 + 회색지대 LLM 판정]")
        for low, high in [(0.50, 0.62), (0.50, 0.65), (0.48, 0.62), (0.52, 0.60)]:
            _llm_calls[0] = 0
            pred = cluster_gray_zone(qs, embed_bge, low, high)
            m = pairwise_metrics(pred, [q["group"] for q in qs])
            print(f"  low={low:.2f} high={high:.2f}  clusters={len(pred):2d}  "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                  f"(FP={m['fp']} FN={m['fn']}, LLM호출={_llm_calls[0]}회)")
        # 최적 밴드로 상세 리포트
        _llm_calls[0] = 0
        pred = cluster_gray_zone(qs, embed_bge, 0.50, 0.62)
        run("bge-m3 + 회색지대 LLM (low=0.50, high=0.62)", pred, qs)
        print(f"  (LLM 호출 총 {_llm_calls[0]}회 — 전체 {len(qs)}개 질문 중 경계 케이스만)")
    except Exception as e:
        import traceback
        print(f"  [skip] 회색지대 LLM 측정 불가: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
