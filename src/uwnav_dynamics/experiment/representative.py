"""
模块名称：代表性样例选择工具

模块职责：
为离线评估与长序列 replay 统一提供“代表性样例/片段”筛选能力，
避免可视化与论文图包继续依赖“前 N 个样本”这种不稳定策略。

主要功能：
1. 定义 `min / max / median` 三种代表性选择规则。
2. 在多条规则下做去重选择，保证 best / median / worst 等样例尽量覆盖不同误差形态。
3. 当规则数量少于需求样例数时，用主规则自动补齐剩余样例。

数据流：
样例级指标 rows
    ↓
representative rules
    ↓
去重排序与补齐
    ↓
带 `representative_*` 元信息的 rows

依赖模块：
- dataclasses
- math

备注：
- 本模块只负责“选哪些样例更有代表性”，不负责数值评估与文件落盘。
- 输入 rows 应该已经包含样例 id 和候选排序所需的数值指标。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RepresentativeRule:
    """单条代表性样例选择规则。"""
    tag: str
    metric: str
    mode: str  # "min" | "max" | "median"


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _finite_values(rows: Sequence[Mapping[str, Any]], *, metric: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        val = _safe_float(row.get(metric))
        if math.isfinite(val):
            out.append(val)
    return out


def _sorted_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_key: str,
    metric: str,
    mode: str,
) -> list[Mapping[str, Any]]:
    if mode not in {"min", "max", "median"}:
        raise ValueError(f"unsupported representative mode: {mode!r}")

    if mode == "median":
        finite_vals = _finite_values(rows, metric=metric)
        target = sorted(finite_vals)[len(finite_vals) // 2] if finite_vals else float("nan")
        return sorted(
            rows,
            key=lambda row: (
                not math.isfinite(_safe_float(row.get(metric))),
                abs(_safe_float(row.get(metric)) - target) if math.isfinite(target) else float("inf"),
                str(row.get(id_key, "")),
            ),
        )

    if mode == "min":
        return sorted(
            rows,
            key=lambda row: (
                not math.isfinite(_safe_float(row.get(metric))),
                _safe_float(row.get(metric)) if math.isfinite(_safe_float(row.get(metric))) else float("inf"),
                str(row.get(id_key, "")),
            ),
        )

    return sorted(
        rows,
        key=lambda row: (
            not math.isfinite(_safe_float(row.get(metric))),
            -_safe_float(row.get(metric)) if math.isfinite(_safe_float(row.get(metric))) else float("inf"),
            str(row.get(id_key, "")),
        ),
    )


def select_representative_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_key: str,
    max_items: int,
    rules: Sequence[RepresentativeRule],
) -> list[dict[str, Any]]:
    """
    根据多条规则从样例级指标中选择代表性行，并写入代表性标签。
    """
    if max_items <= 0 or len(rows) <= 0:
        return []
    if len(rules) <= 0:
        raise ValueError("representative rules must be non-empty")

    selected_ids: set[str] = set()
    selected_rows: list[dict[str, Any]] = []

    def _append_first_available(candidates: Iterable[Mapping[str, Any]], *, rule: RepresentativeRule) -> None:
        nonlocal selected_rows
        for row in candidates:
            row_id = str(row.get(id_key, ""))
            if row_id == "" or row_id in selected_ids:
                continue
            payload = dict(row)
            payload["representative_tag"] = str(rule.tag)
            payload["representative_metric"] = str(rule.metric)
            payload["representative_mode"] = str(rule.mode)
            payload["representative_value"] = _safe_float(row.get(rule.metric))
            selected_rows.append(payload)
            selected_ids.add(row_id)
            return

    for rule in rules:
        if len(selected_rows) >= int(max_items):
            break
        _append_first_available(
            _sorted_candidates(rows, id_key=id_key, metric=rule.metric, mode=rule.mode),
            rule=rule,
        )

    if len(selected_rows) < int(max_items):
        primary = rules[0]
        extras = _sorted_candidates(rows, id_key=id_key, metric=primary.metric, mode=primary.mode)
        extra_idx = 0
        for row in extras:
            if len(selected_rows) >= int(max_items):
                break
            row_id = str(row.get(id_key, ""))
            if row_id == "" or row_id in selected_ids:
                continue
            payload = dict(row)
            payload["representative_tag"] = f"extra_{extra_idx:02d}"
            payload["representative_metric"] = str(primary.metric)
            payload["representative_mode"] = str(primary.mode)
            payload["representative_value"] = _safe_float(row.get(primary.metric))
            selected_rows.append(payload)
            selected_ids.add(row_id)
            extra_idx += 1

    return selected_rows
