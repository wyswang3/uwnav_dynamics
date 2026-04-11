"""
模块名称：代表性样例选择测试

模块职责：
验证代表性样例选择工具会按 `best / median / worst` 等规则稳定选出唯一候选，
避免评估与 replay 图继续依赖“前 N 个样例”。

主要功能：
1. 检查 `min / median / max` 三类规则是否按预期排序。
2. 检查多规则组合下的去重与补齐逻辑。
"""

from __future__ import annotations

from uwnav_dynamics.experiment.representative import RepresentativeRule, select_representative_rows


def test_select_representative_rows_marks_best_median_worst_uniquely() -> None:
    rows = [
        {"sample_slot": 0, "rmse_global": 0.10, "final_step_rmse": 0.10},
        {"sample_slot": 1, "rmse_global": 0.20, "final_step_rmse": 0.20},
        {"sample_slot": 2, "rmse_global": 0.30, "final_step_rmse": 0.50},
        {"sample_slot": 3, "rmse_global": 0.40, "final_step_rmse": 0.40},
        {"sample_slot": 4, "rmse_global": 0.50, "final_step_rmse": 0.30},
    ]

    picked = select_representative_rows(
        rows,
        id_key="sample_slot",
        max_items=4,
        rules=(
            RepresentativeRule(tag="best_rmse", metric="rmse_global", mode="min"),
            RepresentativeRule(tag="median_rmse", metric="rmse_global", mode="median"),
            RepresentativeRule(tag="worst_rmse", metric="rmse_global", mode="max"),
            RepresentativeRule(tag="worst_final_step", metric="final_step_rmse", mode="max"),
        ),
    )

    assert [row["representative_tag"] for row in picked] == [
        "best_rmse",
        "median_rmse",
        "worst_rmse",
        "worst_final_step",
    ]
    assert [int(row["sample_slot"]) for row in picked] == [0, 2, 4, 3]


def test_select_representative_rows_fills_extra_slots_with_primary_rule() -> None:
    rows = [
        {"segment_id": 10, "rmse_global": 0.10},
        {"segment_id": 11, "rmse_global": 0.20},
        {"segment_id": 12, "rmse_global": 0.30},
    ]

    picked = select_representative_rows(
        rows,
        id_key="segment_id",
        max_items=3,
        rules=(RepresentativeRule(tag="best_rmse", metric="rmse_global", mode="min"),),
    )

    assert [row["representative_tag"] for row in picked] == [
        "best_rmse",
        "extra_00",
        "extra_01",
    ]
    assert [int(row["segment_id"]) for row in picked] == [10, 11, 12]
