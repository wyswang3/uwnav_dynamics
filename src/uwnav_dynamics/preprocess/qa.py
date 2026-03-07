"""
模块名称：预处理产物 QA / 验收

模块职责：
对 `train_base.csv` 这类预处理产物执行最小但实用的质量审查，
在进入 dataset build 或训练前尽早发现时间轴、dense target 与窗口可构造性问题。

主要功能：
1. 提供 hard checks：时间列、dense target finite、最小行数等 fail-fast 校验。
2. 提供 soft checks：DVL / Power 覆盖率与缺失列 warning。
3. 输出统一的 PASS / FAIL / WARNING 摘要与关键列基础统计。

数据流：
train_base.csv / aligned DataFrame
    ↓
run_train_base_qa()
    ↓
TrainBaseQaReport
    ↓
render_train_base_qa() / assert_train_base_qa_pass()

依赖模块：
- numpy
- pandas
- dataclasses

备注：
- 本模块只做检查与摘要，不做任何静默修复；
- 若 hard checks 失败，应回到上游 align / preprocess 重新生成产物。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QaIssue:
    level: str  # "error" | "warning"
    code: str
    message: str


@dataclass(frozen=True)
class QaCoverage:
    name: str
    col: str
    ratio: float
    n_valid: int
    n_total: int


@dataclass(frozen=True)
class QaNumericStat:
    col: str
    n_total: int
    n_nonfinite: int
    mean: float
    std: float
    min: float
    max: float


@dataclass(frozen=True)
class TrainBaseQaReport:
    stage: str
    n_rows: int
    time_col: str
    hist_len: Optional[int]
    pred_len: Optional[int]
    issues: tuple[QaIssue, ...]
    coverages: tuple[QaCoverage, ...]
    stats: tuple[QaNumericStat, ...]

    @property
    def hard_error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "warning")

    @property
    def passed(self) -> bool:
        return self.hard_error_count == 0


def _append_issue(issues: list[QaIssue], *, level: str, code: str, message: str) -> None:
    issues.append(QaIssue(level=level, code=code, message=message))


def _find_first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _compute_numeric_stat(df: pd.DataFrame, col: str) -> QaNumericStat:
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(arr)
    n_total = int(arr.size)
    n_nonfinite = int((~finite).sum())

    if finite.any():
        finite_arr = arr[finite]
        mean = float(np.mean(finite_arr))
        std = float(np.std(finite_arr))
        min_v = float(np.min(finite_arr))
        max_v = float(np.max(finite_arr))
    else:
        mean = float("nan")
        std = float("nan")
        min_v = float("nan")
        max_v = float("nan")

    return QaNumericStat(
        col=col,
        n_total=n_total,
        n_nonfinite=n_nonfinite,
        mean=mean,
        std=std,
        min=min_v,
        max=max_v,
    )


def run_train_base_qa(
    df: pd.DataFrame,
    *,
    stage: str,
    time_col: str,
    dense_target_cols: Sequence[str],
    hist_len: Optional[int] = None,
    pred_len: Optional[int] = None,
    key_stat_cols: Optional[Sequence[str]] = None,
    dvl_mask_candidates: Sequence[str] = ("dvl_mask", "has_dvl"),
    power_mask_candidates: Sequence[str] = ("power_mask", "has_power"),
) -> TrainBaseQaReport:
    """
    对 train_base / base_csv DataFrame 执行最小 QA。

    hard checks：
      - 时间列存在、finite、严格递增
      - dense target 列存在且全 finite
      - 行数足够构造滑窗（若提供 hist_len / pred_len）

    soft checks：
      - DVL / Power 覆盖率统计
      - 覆盖率为 0 或 mask 列缺失时给 warning
      - 关键列基础统计
    """
    issues: list[QaIssue] = []
    coverages: list[QaCoverage] = []
    stats: list[QaNumericStat] = []

    n_rows = int(len(df))

    if time_col not in df.columns:
        _append_issue(
            issues,
            level="error",
            code="missing_time_col",
            message=f"time column {time_col!r} not found in {stage}",
        )
    else:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        n_bad_t = int((~np.isfinite(t)).sum())
        if n_bad_t > 0:
            _append_issue(
                issues,
                level="error",
                code="time_nonfinite",
                message=f"{stage} time column {time_col!r} has {n_bad_t} non-finite values",
            )
        bad_steps = np.where(np.diff(t) <= 0.0)[0]
        if bad_steps.size > 0:
            i = int(bad_steps[0])
            _append_issue(
                issues,
                level="error",
                code="time_not_strictly_increasing",
                message=(
                    f"{stage} time column {time_col!r} must be strictly increasing: "
                    f"bad_idx={i} t[i]={float(t[i]):.9f} t[i+1]={float(t[i + 1]):.9f}"
                ),
            )

    span = None
    if hist_len is not None and pred_len is not None:
        span = int(hist_len) + int(pred_len)
        if n_rows < span:
            _append_issue(
                issues,
                level="error",
                code="insufficient_rows_for_windows",
                message=(
                    f"{stage} rows not enough for sliding window: "
                    f"n_rows={n_rows}, required>={span} (hist_len={int(hist_len)}, pred_len={int(pred_len)})"
                ),
            )

    missing_targets = [c for c in dense_target_cols if c not in df.columns]
    if missing_targets:
        _append_issue(
            issues,
            level="error",
            code="missing_target_cols",
            message=f"{stage} missing target columns: {missing_targets}",
        )

    bad_target_parts: list[str] = []
    for c in dense_target_cols:
        if c not in df.columns:
            continue
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad > 0:
            bad_target_parts.append(f"{c}={n_bad}")

    if bad_target_parts:
        _append_issue(
            issues,
            level="error",
            code="nonfinite_dense_targets",
            message=(
                f"non-finite target cols in {stage}: "
                + ", ".join(bad_target_parts)
                + ". Please re-run align / rebuild dataset."
            ),
        )

    for coverage_name, candidates in (
        ("dvl_mask", dvl_mask_candidates),
        ("power_mask", power_mask_candidates),
    ):
        col = _find_first_existing_col(df, candidates)
        if col is None:
            _append_issue(
                issues,
                level="warning",
                code=f"missing_{coverage_name}_col",
                message=f"{stage} missing coverage column for {coverage_name}: candidates={tuple(candidates)}",
            )
            continue

        mask_arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(mask_arr) & (mask_arr > 0.5)
        n_valid = int(valid.sum())
        n_total = int(mask_arr.size)
        ratio = float(valid.mean()) if n_total > 0 else 0.0
        coverages.append(
            QaCoverage(
                name=coverage_name,
                col=col,
                ratio=ratio,
                n_valid=n_valid,
                n_total=n_total,
            )
        )
        if ratio <= 0.0:
            _append_issue(
                issues,
                level="warning",
                code=f"zero_{coverage_name}_coverage",
                message=f"{stage} {coverage_name} coverage is zero (col={col!r})",
            )

    stat_cols: list[str] = []
    for c in [time_col, *(key_stat_cols if key_stat_cols is not None else dense_target_cols)]:
        if c in df.columns and c not in stat_cols:
            stat_cols.append(c)
    for c in stat_cols:
        stats.append(_compute_numeric_stat(df, c))

    return TrainBaseQaReport(
        stage=stage,
        n_rows=n_rows,
        time_col=time_col,
        hist_len=None if hist_len is None else int(hist_len),
        pred_len=None if pred_len is None else int(pred_len),
        issues=tuple(issues),
        coverages=tuple(coverages),
        stats=tuple(stats),
    )


def render_train_base_qa(report: TrainBaseQaReport) -> str:
    """
    渲染统一 QA 摘要，便于 CLI / pipeline 直接打印。
    """
    status = "PASS" if report.passed else "FAIL"
    span_desc = ""
    if report.hist_len is not None and report.pred_len is not None:
        span_desc = (
            f" hist_len={report.hist_len} pred_len={report.pred_len}"
            f" total_span={report.hist_len + report.pred_len}"
        )

    lines = [
        f"[QA][{report.stage}][SUMMARY] status={status} "
        f"hard={report.hard_error_count} warn={report.warning_count} "
        f"rows={report.n_rows} time_col={report.time_col}{span_desc}"
    ]

    for issue in report.issues:
        tag = "ERROR" if issue.level == "error" else "WARN"
        lines.append(f"[QA][{report.stage}][{tag}] code={issue.code} msg={issue.message}")

    for cov in report.coverages:
        lines.append(
            f"[QA][{report.stage}][COVERAGE] name={cov.name} col={cov.col} "
            f"ratio={cov.ratio:.6f} valid={cov.n_valid} total={cov.n_total}"
        )

    for stat in report.stats:
        lines.append(
            f"[QA][{report.stage}][STAT] col={stat.col} "
            f"nonfinite={stat.n_nonfinite} mean={stat.mean:.6f} std={stat.std:.6f} "
            f"min={stat.min:.6f} max={stat.max:.6f}"
        )

    return "\n".join(lines)


def assert_train_base_qa_pass(report: TrainBaseQaReport) -> None:
    """
    若存在 hard check 失败，则抛异常中断后续流程。
    """
    if report.passed:
        return

    errors = [issue.message for issue in report.issues if issue.level == "error"]
    raise ValueError("; ".join(errors))
