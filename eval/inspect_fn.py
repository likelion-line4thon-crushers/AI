"""
회색지대 실험(gpt-4o-mini, 밴드 0.50~0.62)에서 LLM이 '다르다'로 판정해
recall 을 떨어뜨린 쌍(= 합성 라벨상 같은 그룹인데 LLM 다르다)을 원문/cosine 과 함께 기록.
사실만 남기고 판단은 하지 않는다.

실행: .venv/Scripts/python.exe eval/inspect_fn.py   (모델은 .env 기본 = gpt-4o-mini)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import run_eval as E  # noqa: E402

LOW, HIGH = 0.50, 0.62


def main():
    gold_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_questions.jsonl")
    qs = E.load_gold(gold_path)

    clusters = []      # {centroid, rep_content, rep_group, members}
    decisions = []     # 모든 회색지대 LLM 판정 로그

    for idx, q in enumerate(qs):
        emb = E.embed_bge(q["content"])
        best_cos, best_i = -1.0, -1
        for i, c in enumerate(clusters):
            cos = float(emb @ c["centroid"])
            if cos > best_cos:
                best_cos, best_i = cos, i

        join = False
        if best_i >= 0:
            if best_cos >= HIGH:
                join = True
            elif best_cos >= LOW:
                rep = clusters[best_i]
                verdict = E.llm_same(q["content"], rep["rep_content"])
                decisions.append({
                    "A": q["content"],
                    "B": rep["rep_content"],
                    "gold_same": q["group"] == rep["rep_group"],
                    "verdict": verdict,     # True=같다, False=다르다
                    "cos": round(best_cos, 3),
                })
                join = verdict

        if join:
            c = clusters[best_i]
            c["members"].append(idx)
            n = len(c["members"])
            c["centroid"] = (c["centroid"] * (n - 1) + emb) / n
        else:
            clusters.append({
                "centroid": emb,
                "rep_content": q["content"],
                "rep_group": q["group"],
                "members": [idx],
            })

    # LLM 이 '다르다'로 판정해서 recall 손해가 된 쌍: 라벨상 같은 그룹인데 verdict=False
    fn_pairs = [d for d in decisions if d["gold_same"] and not d["verdict"]]

    out = {
        "model": E._openai_model,
        "band": [LOW, HIGH],
        "total_llm_calls": len(decisions),
        "gold_same_but_llm_diff": len(fn_pairs),
        "pairs": fn_pairs,
        "all_decisions": decisions,
    }
    safe_model = E._openai_model.replace("/", "-")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"fn_pairs_{safe_model}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("DONE", out["model"], "calls=", out["total_llm_calls"],
          "gold_same_but_diff=", out["gold_same_but_llm_diff"])


if __name__ == "__main__":
    main()
