"""
배포 감각 측정 (밴드 0.50~0.62, gpt-4o):
 1) 전체쌍 중 회색지대 밴드에 드는 비율
 2) 증분 시나리오에서 요청(새 질문)당 gpt-4o 호출 수 (최대 1콜)
 3) gpt-4o 판정 1콜당 응답 지연(레이턴시)

실행: OPENAI_MODEL=gpt-4o .venv/Scripts/python.exe eval/measure_deploy.py
"""
import json
import os
import statistics
import sys
import time
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import run_eval as E  # noqa: E402
from config.settings import settings  # noqa: E402
from openai import OpenAI  # noqa: E402

LOW, HIGH = 0.50, 0.62
_client = OpenAI(api_key=settings.OPENAI_API_KEY)
_MODEL = settings.OPENAI_MODEL
_latencies = []


def timed_judge(a, b):
    """llm_same 과 동일한 프롬프트로 판정하되 API 왕복 시간을 기록 (캐시 없이 매번 호출)."""
    t0 = time.perf_counter()
    resp = _client.chat.completions.create(
        model=_MODEL,
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
    _latencies.append(time.perf_counter() - t0)
    try:
        return bool(json.loads(resp.choices[0].message.content).get("same", False))
    except Exception:
        return False


def main():
    lines = []
    gold_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_questions.jsonl")
    qs = E.load_gold(gold_path)
    embs = [E.embed_bge(q["content"]) for q in qs]
    n = len(qs)

    # 1) 전체쌍 분포
    total = in_band = auto_join = auto_new = 0
    for i, j in combinations(range(n), 2):
        cos = float(embs[i] @ embs[j])
        total += 1
        if cos >= HIGH:
            auto_join += 1
        elif cos >= LOW:
            in_band += 1
        else:
            auto_new += 1
    lines.append(f"[모델] {_MODEL}  [밴드] [{LOW}, {HIGH})")
    lines.append(
        f"[전체쌍 {total}] 밴드내={in_band} ({in_band/total*100:.1f}%) | "
        f">=HIGH(자동합류)={auto_join} ({auto_join/total*100:.1f}%) | "
        f"<LOW(자동신규)={auto_new} ({auto_new/total*100:.1f}%)"
    )

    # 2) 증분 시나리오: 새 질문마다 best 후보 1개와 비교, band이면 gpt-4o 1콜
    clusters = []
    for idx, q in enumerate(qs):
        emb = embs[idx]
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
                join = timed_judge(q["content"], clusters[best_i]["rep"])
        if join:
            c = clusters[best_i]
            c["members"].append(idx)
            k = len(c["members"])
            c["centroid"] = (c["centroid"] * (k - 1) + emb) / k
        else:
            clusters.append({"centroid": emb, "rep": q["content"], "members": [idx]})

    calls = len(_latencies)
    lines.append(
        f"[증분 호출] 질문 {n}개 처리 중 gpt-4o 호출 {calls}회 "
        f"= 요청당 평균 {calls/n:.2f}회 (요청당 최대 1회)"
    )
    lines.append(
        f"[호출률] {calls}/{n} = 요청의 {calls/n*100:.0f}% 가 gpt-4o 1콜, 나머지 {(1-calls/n)*100:.0f}% 는 0콜"
    )

    # 3) 지연
    if _latencies:
        s = sorted(_latencies)
        p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
        lines.append(
            f"[지연/콜] mean={statistics.mean(_latencies)*1000:.0f}ms "
            f"median={statistics.median(_latencies)*1000:.0f}ms "
            f"min={min(_latencies)*1000:.0f}ms max={max(_latencies)*1000:.0f}ms "
            f"p95={p95*1000:.0f}ms  (표본 {calls}콜)"
        )
        lines.append(
            f"[요청 체감] 호출되는 요청은 평균 +{statistics.mean(_latencies)*1000:.0f}ms, "
            f"호출 안 되는 요청은 +0ms (임베딩 자체 지연은 별도)"
        )

    text = "\n".join(lines)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy.txt"),
              "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
