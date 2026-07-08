"""
임베딩 모델 비교 측정 (회색지대 gpt-4o 판정은 고정, 임베딩만 교체).

측정 대상:
  1. KR-SBERT  (snunlp/KR-SBERT-V40K-klueNLI-augSTS)           — 로컬
  2. OpenAI    text-embedding-3-small / text-embedding-3-large — API
  (참고 기준) bge-m3: 코사인-only F1 0.716 / 회색지대 0.852 (별도 측정, gray_4o.txt)

각 모델마다:
  A. 코사인 분포 (같은그룹 vs 다른그룹 퍼센타일) → 회색지대 밴드가 맞는지 진단
  B. 코사인-only 임계값 스윕 → 최적 F1 / 임계값(thr*)
  C. 회색지대 gpt-4o 판정:
       - bge-m3 기본 밴드 (0.50 / 0.62) 그대로 적용 (밴드 미스매치 확인용)
       - 각 모델 분포에 맞춰 조정한 밴드 (thr* 기준 그리드)

실행: 프로젝트 루트에서
    .venv/Scripts/python.exe eval/run_embed_compare.py
"""
import os
import sys
from itertools import combinations

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import run_eval as E  # noqa: E402
from config.settings import settings  # noqa: E402

TS = E.TS


# ── OpenAI 임베딩 (배치 프리컴퓨트 후 조회) ──────────────────────
_openai_emb_cache = {}  # (model, normtext) -> np.ndarray(unit)
_openai_client_emb = None


def _emb_client():
    global _openai_client_emb
    if _openai_client_emb is None:
        from openai import OpenAI
        _openai_client_emb = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client_emb


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def precompute_openai(model: str, qs):
    """gold 질문 전체를 한 번의 API 호출로 임베딩(정규화 원문 기준, 중복 제거)."""
    texts = [TS.normalize(q["content"]) for q in qs]
    uniq = list(dict.fromkeys(texts))
    resp = _emb_client().embeddings.create(model=model, input=uniq)
    for t, d in zip(uniq, resp.data):
        _openai_emb_cache[(model, t)] = _unit(d.embedding)


def make_openai_embed(model: str):
    def embed(text: str) -> np.ndarray:
        t = TS.normalize(text)
        key = (model, t)
        if key not in _openai_emb_cache:  # 안전망: 프리컴퓨트에 없으면 단건 호출
            r = _emb_client().embeddings.create(model=model, input=[t])
            _openai_emb_cache[key] = _unit(r.data[0].embedding)
        return _openai_emb_cache[key]
    return embed


# ── 코사인 분포 진단 ──────────────────────────────────────────
def cosine_distribution(qs, embed):
    embs = [embed(q["content"]) for q in qs]
    groups = [q["group"] for q in qs]
    same, diff = [], []
    for i, j in combinations(range(len(qs)), 2):
        c = float(embs[i] @ embs[j])
        (same if groups[i] == groups[j] else diff).append(c)
    return np.array(same), np.array(diff)


def _pct(a, ps):
    return {p: (float(np.percentile(a, p)) if len(a) else float("nan")) for p in ps}


def print_distribution(name, same, diff):
    ps = [5, 10, 25, 50, 75, 90, 95]
    s = _pct(same, ps)
    d = _pct(diff, ps)
    print(f"\n[코사인 분포 · {name}]  (같은그룹쌍 {len(same)}개 / 다른그룹쌍 {len(diff)}개)")
    print("        p05    p10    p25    p50    p75    p90    p95")
    print("  같음:" + "".join(f" {s[p]:6.3f}" for p in ps))
    print("  다름:" + "".join(f" {d[p]:6.3f}" for p in ps))
    # 밴드 배치 힌트: 다름 상위(=자동합류 상한 후보) / 같음 하위(=자동신규 하한 후보)
    print(f"  → 밴드 힌트: high ≈ 다름p90({d[90]:.3f})~p95({d[95]:.3f}) 위, "
          f"low ≈ 같음p10({s[10]:.3f})~p25({s[25]:.3f}) 근처")


# ── 임계값 스윕 (코사인-only) ─────────────────────────────────
def sweep_cosine_only(name, qs, embed, grid):
    gold_groups = [q["group"] for q in qs]
    print(f"\n[코사인-only 임계값 스윕 · {name}]")
    best = None
    for thr in grid:
        pred = E.cluster_cosine_only(qs, embed, float(thr))
        m = E.pairwise_metrics(pred, gold_groups)
        mark = ""
        if best is None or m["f1"] > best[1]["f1"]:
            best = (float(thr), m, pred)
            mark = "  <= best"
        print(f"  thr={thr:.2f}  clusters={len(pred):2d}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"(FP={m['fp']} FN={m['fn']}){mark}")
    print(f"  → 최적 thr*={best[0]:.2f}, F1={best[1]['f1']:.3f}")
    return best


# ── 회색지대 gpt-4o 판정 ──────────────────────────────────────
def gray_zone_grid(name, qs, embed, thr_star, extra_bands=None):
    gold_groups = [q["group"] for q in qs]
    # bge-m3 기본 밴드 + thr* 기준 조정 밴드
    bands = [(0.50, 0.62)]
    for dl, dh in [(-0.05, +0.07), (-0.05, +0.03), (-0.10, +0.05), (-0.08, 0.00), (-0.10, +0.10)]:
        lo, hi = round(thr_star + dl, 2), round(thr_star + dh, 2)
        if hi > lo and (lo, hi) not in bands:
            bands.append((lo, hi))
    for b in (extra_bands or []):
        if b not in bands:
            bands.append(b)

    print(f"\n[회색지대 gpt-4o 판정 · {name}]  (밴드: bge-m3기본 0.50/0.62 + thr*={thr_star:.2f} 조정)")
    best = None
    for low, high in bands:
        E._llm_calls[0] = 0
        pred = E.cluster_gray_zone(qs, embed, low, high)
        m = E.pairwise_metrics(pred, gold_groups)
        tag = "  (bge-m3 기본밴드)" if (low, high) == (0.50, 0.62) else ""
        mark = ""
        if best is None or m["f1"] > best[1]["f1"]:
            best = ((low, high), m, pred)
            mark = "  <= best"
        print(f"  low={low:.2f} high={high:.2f}  clusters={len(pred):2d}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"(FP={m['fp']} FN={m['fn']}, LLM신규호출={E._llm_calls[0]}회){tag}{mark}")
    print(f"  → 최적 밴드 low={best[0][0]:.2f}/high={best[0][1]:.2f}, F1={best[1]['f1']:.3f}")
    return best


# ── 실행 ──────────────────────────────────────────────────────
def measure_model(name, qs, embed, sweep_grid):
    print("\n" + "=" * 72)
    print(f"■ {name}")
    print("=" * 72)
    same, diff = cosine_distribution(qs, embed)
    print_distribution(name, same, diff)
    best_cos = sweep_cosine_only(name, qs, embed, sweep_grid)
    best_gray = gray_zone_grid(name, qs, embed, best_cos[0])
    return {
        "name": name,
        "cos_thr": best_cos[0], "cos_f1": best_cos[1]["f1"],
        "cos_p": best_cos[1]["precision"], "cos_r": best_cos[1]["recall"],
        "gray_band": best_gray[0], "gray_f1": best_gray[1]["f1"],
        "gray_p": best_gray[1]["precision"], "gray_r": best_gray[1]["recall"],
    }


def main():
    gold_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_questions.jsonl")
    qs = E.load_gold(gold_path)
    print(f"[load] 질문 {len(qs)}개, 정답 group {len(set(q['group'] for q in qs))}개")

    # 판정기 = gpt-4o 고정 (production 회색지대와 동일). 요약모델(gpt-4o-mini)로 새지 않게 선점.
    from openai import OpenAI
    E._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    E._openai_model = "gpt-4o"
    try:
        ok = E.llm_same("발표 자료 보내주세요", "PPT 어디서 받아요?")
        print(f"[check] OpenAI 연결 OK, 판정모델={E._openai_model}, 샘플판정={ok}")
    except Exception as e:
        print(f"[check] OpenAI 호출 실패: {e}")
        return

    results = []

    # 1) KR-SBERT (로컬) — embed_raw 가 곧 KR-SBERT 정규화 원문 임베딩
    st_grid = np.arange(0.40, 0.86, 0.05)
    results.append(measure_model("KR-SBERT", qs, E.embed_raw, st_grid))

    # 2) OpenAI 임베딩 (배치 프리컴퓨트 후 측정)
    oa_grid = np.arange(0.20, 0.81, 0.05)
    for model in ["text-embedding-3-small", "text-embedding-3-large"]:
        try:
            precompute_openai(model, qs)
            results.append(measure_model(f"OpenAI {model}", qs, make_openai_embed(model), oa_grid))
        except Exception as ex:
            import traceback
            print(f"\n[skip] {model} 측정 불가: {ex}")
            traceback.print_exc()

    # ── 최종 비교표 ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("■ 최종 비교표 (회색지대 판정 = gpt-4o 고정)")
    print("=" * 72)
    print(f"{'임베딩':<26} {'코사인-only best':<22} {'회색지대(gpt-4o) best':<26}")
    print(f"{'':<26} {'F1  (thr / P / R)':<22} {'F1  (밴드 / P / R)':<26}")
    print("-" * 72)
    # 참고: bge-m3 (별도 측정값, gray_4o.txt)
    print(f"{'bge-m3 (참고,측정불가서버)':<26} "
          f"{'0.716 (0.55/.73/.71)':<22} {'0.852 (.50-.62/.96/.77)':<26}")
    for r in results:
        cos = f"{r['cos_f1']:.3f} ({r['cos_thr']:.2f}/{r['cos_p']:.2f}/{r['cos_r']:.2f})"
        gb = r["gray_band"]
        gray = f"{r['gray_f1']:.3f} ({gb[0]:.2f}-{gb[1]:.2f}/{r['gray_p']:.2f}/{r['gray_r']:.2f})"
        print(f"{r['name']:<26} {cos:<22} {gray:<26}")
    print("\n주의: bge-m3 기본밴드(0.50/0.62)를 다른 모델에 그대로 쓰면 코사인 스케일이 달라 "
          "회색지대가 거의 작동 안 할 수 있음 → 위 '조정 밴드' 열이 각 모델 맞춤값.")


if __name__ == "__main__":
    main()
