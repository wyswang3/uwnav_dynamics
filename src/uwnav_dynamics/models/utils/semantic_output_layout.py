"""
模块名称：语义输出布局契约

模块职责：
统一管理模型输出 9 维状态的物理语义说明，
供评估指标聚合、artifact metadata 与 viz 分组解释共同复用。

主要功能：
1. 定义当前 canonical 输出语义：`Acc 3 + Gyro 3 + Vel_state 3`。
2. 将语义布局写成最小 metadata，供 `metrics.yaml` 落盘。
3. 从 `metrics.yaml` 读取 semantic metadata，或在旧 artifact 上统一 fallback。
4. 使用 `target_cols` 做语义校验，但不参与主执行路径的布局裁决。

数据流：
canonical output contract / optional target_cols
    ↓
semantic output layout
    ↓
metrics.yaml.layout.semantic
    ↓
eval metric grouping / viz grouping

依赖模块：
- yaml
- warnings

备注：
- 本模块不负责从输入张量中提取 `y0`，执行索引请使用 `execution_layout.py`。
- `target_cols` 仅作校验旁证，不是 train / eval / viz 的第二执行真源。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import warnings

import yaml


SEMANTIC_LAYOUT_SCHEMA_VERSION = "state_layout_v1"
CANONICAL_SEMANTIC_SOURCE = "canonical_acc_gyro_vel_v1"
FALLBACK_WARNING = "missing layout.semantic metadata; fallback to canonical acc/gyro/vel grouping"

_CANONICAL_COMPONENT_LABELS: tuple[str, ...] = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "vel_x",
    "vel_y",
    "vel_z",
)
_CANONICAL_GROUP_INDICES: dict[str, tuple[int, ...]] = {
    "acc": (0, 1, 2),
    "gyro": (3, 4, 5),
    "vel": (6, 7, 8),
}


@dataclass(frozen=True)
class SemanticOutputLayout:
    """模型输出维度的语义标签与分组定义。"""
    source: str
    component_labels: tuple[str, ...]
    group_indices: dict[str, tuple[int, ...]]
    validated_against_target_cols: bool = False


def canonical_semantic_output_layout(dout: int) -> SemanticOutputLayout:
    """返回当前项目默认的 `acc/gyro/vel` 语义输出布局。"""
    if int(dout) != len(_CANONICAL_COMPONENT_LABELS):
        raise ValueError(
            "Current canonical semantic output layout expects dout=9 "
            f"(Acc3 + Gyro3 + Vel3), got dout={dout}"
        )
    return SemanticOutputLayout(
        source=CANONICAL_SEMANTIC_SOURCE,
        component_labels=_CANONICAL_COMPONENT_LABELS,
        group_indices=dict(_CANONICAL_GROUP_INDICES),
        validated_against_target_cols=False,
    )


def resolve_semantic_output_layout(
    *,
    dout: int,
    target_cols: Sequence[str] | None = None,
) -> SemanticOutputLayout:
    """解析输出语义布局，并在可用时用 `target_cols` 做一致性校验。"""
    layout = canonical_semantic_output_layout(dout)
    if target_cols is None:
        return layout
    return validate_target_cols_against_semantic_layout(target_cols, layout)


def build_semantic_layout_metadata(layout: SemanticOutputLayout) -> dict[str, object]:
    """把语义布局对象转换为可落盘的 metadata 字典。"""
    return {
        "source": layout.source,
        "component_labels": list(layout.component_labels),
        "group_indices": {key: list(indices) for key, indices in layout.group_indices.items()},
        "validated_against_target_cols": bool(layout.validated_against_target_cols),
    }


def load_semantic_layout_from_metrics_dict(
    metrics: Mapping[str, Any],
    *,
    dout: int,
    warn_fn: Callable[[str], None] | None = None,
) -> SemanticOutputLayout:
    """从已加载的 `metrics.yaml` 字典中恢复语义布局。"""
    warn = warnings.warn if warn_fn is None else warn_fn
    layout_meta = metrics.get("layout") if isinstance(metrics, Mapping) else None
    if isinstance(layout_meta, Mapping) and "schema_version" in layout_meta:
        schema_version = str(layout_meta["schema_version"])
        if schema_version != SEMANTIC_LAYOUT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported layout schema_version in metrics.yaml: "
                f"{schema_version!r} != {SEMANTIC_LAYOUT_SCHEMA_VERSION!r}"
            )
    semantic_meta = layout_meta.get("semantic") if isinstance(layout_meta, Mapping) else None
    if not isinstance(semantic_meta, Mapping):
        warn(FALLBACK_WARNING)
        return canonical_semantic_output_layout(dout)

    component_labels = tuple(str(x) for x in semantic_meta.get("component_labels", ()))
    group_indices_raw = semantic_meta.get("group_indices", {})
    if len(component_labels) != int(dout) or not isinstance(group_indices_raw, Mapping):
        raise ValueError("Invalid layout.semantic metadata structure in metrics.yaml")

    group_indices: dict[str, tuple[int, ...]] = {}
    for key, indices in group_indices_raw.items():
        if not isinstance(indices, Sequence):
            raise TypeError(f"layout.semantic.group_indices[{key!r}] must be a sequence")
        group_indices[str(key)] = tuple(int(i) for i in indices)

    return SemanticOutputLayout(
        source=str(semantic_meta.get("source", CANONICAL_SEMANTIC_SOURCE)),
        component_labels=component_labels,
        group_indices=group_indices,
        validated_against_target_cols=bool(semantic_meta.get("validated_against_target_cols", False)),
    )


def load_semantic_layout_from_metrics_path(
    metrics_path: Path,
    *,
    dout: int,
    warn_fn: Callable[[str], None] | None = None,
) -> SemanticOutputLayout:
    """从 `metrics.yaml` 文件路径恢复语义布局，缺失时回退 canonical 布局。"""
    if not metrics_path.exists():
        warn = warnings.warn if warn_fn is None else warn_fn
        warn(FALLBACK_WARNING)
        return canonical_semantic_output_layout(dout)
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = yaml.safe_load(f) or {}
    return load_semantic_layout_from_metrics_dict(metrics, dout=dout, warn_fn=warn_fn)


def validate_target_cols_against_semantic_layout(
    target_cols: Sequence[str],
    layout: SemanticOutputLayout,
) -> SemanticOutputLayout:
    """校验 `target_cols` 的语义顺序与既定布局完全一致。"""
    normalized = tuple(_normalize_target_col_name(col) for col in target_cols)
    if len(normalized) != len(layout.component_labels):
        raise ValueError(
            "target_cols length does not match semantic output layout: "
            f"{len(normalized)} vs {len(layout.component_labels)}"
        )
    if normalized != layout.component_labels:
        raise ValueError(
            "target_cols semantic order mismatch: "
            f"expect {layout.component_labels}, got {normalized}"
        )
    return SemanticOutputLayout(
        source=layout.source,
        component_labels=layout.component_labels,
        group_indices=dict(layout.group_indices),
        validated_against_target_cols=True,
    )


def _normalize_target_col_name(col: str) -> str:
    lowered = str(col).strip().lower()
    canonical = (
        lowered
        .replace("kf", "")
        .replace("state", "")
        .replace("__", "_")
    )
    prefix: str
    axis: str
    if "accx" in canonical:
        prefix, axis = "acc", "x"
    elif "accy" in canonical:
        prefix, axis = "acc", "y"
    elif "accz" in canonical:
        prefix, axis = "acc", "z"
    elif "gyrox" in canonical:
        prefix, axis = "gyro", "x"
    elif "gyroy" in canonical:
        prefix, axis = "gyro", "y"
    elif "gyroz" in canonical:
        prefix, axis = "gyro", "z"
    elif "velx" in canonical:
        prefix, axis = "vel", "x"
    elif "vely" in canonical:
        prefix, axis = "vel", "y"
    elif "velz" in canonical:
        prefix, axis = "vel", "z"
    else:
        raise ValueError(f"Unsupported target_cols semantic label: {col!r}")
    return f"{prefix}_{axis}"
