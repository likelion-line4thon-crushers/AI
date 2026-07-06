"""
회색지대 판정의 '상한선' 측정 (OpenAI 없이).

LLM 대신 정답 group 라벨로 완벽 판정하는 oracle 을 회색지대에 넣어,
'경계를 완벽히 맞춘다면 F1 이 얼마까지 오르는지' = Phase 2 의 상한선을 본다.
bge-m3 코사인만(F1 0.716) 대비 얼마나 개선 여지가 있는지 판단용.

실행: .venv/Scripts/python.exe eval/run_gray_oracle.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.run_eval import (  # noqa: E402
    load_gold, embed_bge, cluster_gray_zone, cluster_cosine_only,
    pairwise_metrics, run,
)


def main():
    gold_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_questions.jsonl")
    qs = load_gold(gold_path)
    group_of = {q["content"]: q["group"] for q in qs}

    def oracle(a: str, b: str) -> bool:
        # 정답 라벨 기반 완벽 판정 (LLM 상한선 시뮬레이션)
        return group_of.get(a) == group_of.get(b)

    gold_groups = [q["group"] for q in qs]

    # 기준선: bge-m3 코사인만 최적
    pred = cluster_cosine_only(qs, embed_bge, 0.55)
    run("[기준] bge-m3 코사인만 (thr=0.55)", pred, qs)

    print("\n[bge-m3 + 회색지대 ORACLE(완벽판정) · 상한선]")
    for low, high in [(0.50, 0.62), (0.48, 0.65), (0.45, 0.65), (0.45, 0.70), (0.40, 0.75)]:
        pred = cluster_gray_zone(qs, embed_bge, low, high, judge=oracle)
        m = pairwise_metrics(pred, gold_groups)
        print(f"  low={low:.2f} high={high:.2f}  clusters={len(pred):2d}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"(FP={m['fp']} FN={m['fn']})")

    pred = cluster_gray_zone(qs, embed_bge, 0.45, 0.70, judge=oracle)
    run("bge-m3 + 회색지대 ORACLE (low=0.45, high=0.70)", pred, qs)


if __name__ == "__main__":
    main()
