"""
eval/run_eval.py 의 pairwise_metrics 계산 검증.
"""
from eval.run_eval import pairwise_metrics


def test_pairwise_metrics_counts_tp_fn_correctly():
    """정답 A A B B 에서 A쌍만 묶고 B쌍은 놓친 경우 TP/FP/FN 과 P/R/F1 이 정확하다."""
    gold = ["A", "A", "B", "B"]
    pred = [[0, 1], [2], [3]]  # (0,1)만 한 클러스터, 2·3은 각각 싱글턴
    m = pairwise_metrics(pred, gold)

    # same-gold 쌍: (0,1)=A, (2,3)=B → 2개 / same-pred 쌍: (0,1) → 1개
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 1)
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert abs(m["f1"] - (2 * 1.0 * 0.5 / (1.0 + 0.5))) < 1e-9


def test_pairwise_metrics_false_merge_counts_fp():
    """서로 다른 그룹(A,B)을 한 클러스터로 묶으면 FP로 잡힌다."""
    gold = ["A", "B"]
    pred = [[0, 1]]  # 오합류
    m = pairwise_metrics(pred, gold)

    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 0)
    assert m["precision"] == 0.0
    # TP+FN=0 → recall 은 폴백 1.0
    assert m["recall"] == 1.0


def test_pairwise_metrics_all_singletons_fallback():
    """전부 싱글턴이면 precision=1.0, recall=0, f1=0 폴백이 적용된다."""
    gold = ["A", "A", "B"]
    pred = [[0], [1], [2]]  # 아무것도 묶지 않음
    m = pairwise_metrics(pred, gold)

    # same-pred 쌍 없음 → TP+FP=0 → precision 폴백 1.0
    # same-gold 쌍 (0,1)=A → FN=1 → recall 0
    assert m["precision"] == 1.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
