"""
bge-m3 + 회색지대 실제 LLM(OpenAI) 판정 측정.
실행: .venv/Scripts/python.exe eval/run_gray_llm.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import run_eval as E  # noqa: E402


def main():
    gold_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_questions.jsonl")
    qs = E.load_gold(gold_path)
    gold_groups = [q["group"] for q in qs]

    # 먼저 API 연결 확인 (1콜)
    try:
        ok = E.llm_same("발표 자료 보내주세요", "PPT 어디서 받아요?")
        print(f"[check] OpenAI 연결 OK, 샘플판정(같은질문?)={ok}, 모델={E._openai_model}")
    except Exception as e:
        print(f"[check] OpenAI 호출 실패: {e}")
        return

    # 기준선
    pred = E.cluster_cosine_only(qs, E.embed_bge, 0.55)
    E.run("[기준] bge-m3 코사인만 (thr=0.55)", pred, qs)

    print("\n[bge-m3 + 회색지대 실제 LLM 판정]")
    for low, high in [(0.50, 0.62), (0.48, 0.65), (0.45, 0.70)]:
        E._llm_cache.clear()
        E._llm_calls[0] = 0
        pred = E.cluster_gray_zone(qs, E.embed_bge, low, high)  # judge=llm_same 기본
        m = E.pairwise_metrics(pred, gold_groups)
        print(f"  low={low:.2f} high={high:.2f}  clusters={len(pred):2d}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"(FP={m['fp']} FN={m['fn']}, LLM호출={E._llm_calls[0]}회)")

    E._llm_cache.clear()
    E._llm_calls[0] = 0
    pred = E.cluster_gray_zone(qs, E.embed_bge, 0.50, 0.62)
    E.run("bge-m3 + 회색지대 실제 LLM (low=0.50, high=0.62)", pred, qs)
    print(f"  (LLM 호출 총 {E._llm_calls[0]}회 / 전체 {len(qs)}개 질문)")


if __name__ == "__main__":
    main()
