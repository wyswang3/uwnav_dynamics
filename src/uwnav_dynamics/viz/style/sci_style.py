"""
模块名称：科研绘图全局风格真源

模块职责：
统一管理仓库内所有科研绘图的全局视觉 token、figure preset、
语义配色、视觉层级与导出规则，避免不同脚本各自发明风格。

主要功能：
1. 提供 `paper` / `ppt` 两类全局绘图 preset。
2. 提供 Acc/Gyro/Vel、X/Y/Z、Observed/Prediction、Primary/Baseline/Ablation 的稳定视觉编码。
3. 提供坐标轴、图例、共享 x 轴与统一导出 helper，供 `viz/eval/*` 与 `viz/plots/*` 复用。

数据流：
绘图脚本选择 preset 与语义 token
    ↓
`setup_mpl()` 设置全局 rcParams
    ↓
plot 模块通过 helper 构造 figure / axes / legend
    ↓
统一导出为 png/pdf，供论文、PPT 与实验审查使用

依赖模块：
- matplotlib
- cycler

备注：
- 本模块是绘图系统的全局真源；`imu_style.py` 只负责多行传感器图布局。
- 主结果与 baseline 的视觉层级必须在此处固化，不应由调用者每次手工拼接样式。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from cycler import cycler


@dataclass(frozen=True)
class FigurePreset:
    """基础画布宽高配置。"""
    width: float
    height: float


@dataclass(frozen=True)
class SeriesStyle:
    """单条曲线的颜色、线型和层级样式。"""
    color: str
    linestyle: str = "-"
    linewidth: float = 1.2
    alpha: float = 1.0
    zorder: float = 3.0


@dataclass(frozen=True)
class MplGlobalStyle:
    """全局 matplotlib 风格 token 集合。"""
    preset: str
    font_family: str
    font_serif: Tuple[str, ...]
    base_fontsize: int
    labelsize: int
    ticksize: int
    legendsize: int
    axes_linewidth: float
    tick_length: float
    tick_width: float
    default_linewidth: float
    grid: bool
    grid_alpha: float
    figure_dpi: int
    savefig_dpi: int
    single_size: Tuple[float, float]
    wide_size: Tuple[float, float]
    sensor_3row_size: Tuple[float, float]
    sensor_2row_size: Tuple[float, float]
    sensor_4x2_size: Tuple[float, float]
    compare_3row_size: Tuple[float, float]
    rollout_3row_size: Tuple[float, float]
    component_3x3_size: Tuple[float, float]
    prop_cycle: Tuple[str, ...]
    legend_frameon: bool
    legend_borderaxespad: float


_STYLE_PRESETS: Dict[str, MplGlobalStyle] = {
    "paper": MplGlobalStyle(
        preset="paper",
        font_family="serif",
        font_serif=("Times New Roman", "Times", "DejaVu Serif"),
        base_fontsize=11,
        labelsize=11,
        ticksize=10,
        legendsize=10,
        axes_linewidth=0.9,
        tick_length=4.0,
        tick_width=0.9,
        default_linewidth=1.2,
        grid=False,
        grid_alpha=0.18,
        figure_dpi=150,
        savefig_dpi=300,
        single_size=(4.7, 3.2),
        wide_size=(6.4, 2.8),
        sensor_3row_size=(4.9, 4.2),
        sensor_2row_size=(4.9, 3.2),
        sensor_4x2_size=(6.2, 5.2),
        compare_3row_size=(5.8, 4.8),
        rollout_3row_size=(5.0, 4.0),
        component_3x3_size=(6.5, 5.4),
        prop_cycle=(
            "#4C78A8",
            "#54A24B",
            "#B279A2",
            "#9D7660",
            "#7F8C8D",
            "#72B7B2",
        ),
        legend_frameon=False,
        legend_borderaxespad=0.25,
    ),
    "ppt": MplGlobalStyle(
        preset="ppt",
        font_family="serif",
        font_serif=("Times New Roman", "Times", "DejaVu Serif"),
        base_fontsize=13,
        labelsize=13,
        ticksize=12,
        legendsize=12,
        axes_linewidth=1.1,
        tick_length=5.0,
        tick_width=1.0,
        default_linewidth=1.6,
        grid=False,
        grid_alpha=0.18,
        figure_dpi=160,
        savefig_dpi=300,
        single_size=(6.1, 4.2),
        wide_size=(7.2, 3.4),
        sensor_3row_size=(6.3, 5.4),
        sensor_2row_size=(6.3, 4.1),
        sensor_4x2_size=(7.6, 6.4),
        compare_3row_size=(7.0, 5.9),
        rollout_3row_size=(6.5, 5.0),
        component_3x3_size=(8.0, 6.6),
        prop_cycle=(
            "#4C78A8",
            "#54A24B",
            "#B279A2",
            "#9D7660",
            "#7F8C8D",
            "#72B7B2",
        ),
        legend_frameon=False,
        legend_borderaxespad=0.25,
    ),
}


_XYZ_SERIES: Dict[str, SeriesStyle] = {
    "x": SeriesStyle(color="#4C78A8"),
    "y": SeriesStyle(color="#F28E2B"),
    "z": SeriesStyle(color="#59A14F"),
}

_GROUP_SERIES: Dict[str, SeriesStyle] = {
    "Acc": SeriesStyle(color="#4C78A8"),
    "Gyro": SeriesStyle(color="#B279A2"),
    "Vel": SeriesStyle(color="#59A14F"),
}

_OBSERVED_PRED_SERIES: Dict[str, SeriesStyle] = {
    "observed": SeriesStyle(color="#30343A", linestyle="-", linewidth=1.6, alpha=1.0, zorder=5),
    "pred": SeriesStyle(color="#1F4E79", linestyle="--", linewidth=1.5, alpha=0.98, zorder=4),
}

_MODEL_ROLE_SERIES: Dict[str, SeriesStyle] = {
    "primary": SeriesStyle(color="#1F4E79", linestyle="-", linewidth=1.8, alpha=1.0, zorder=5),
    "baseline": SeriesStyle(color="#8C9199", linestyle="--", linewidth=1.2, alpha=0.95, zorder=3),
    "ablation": SeriesStyle(color="#5A7D6D", linestyle="-.", linewidth=1.35, alpha=0.98, zorder=4),
}


def get_style(preset: str = "paper") -> MplGlobalStyle:
    """按 preset 名称返回全局绘图风格对象。"""
    if preset not in _STYLE_PRESETS:
        raise KeyError(f"Unknown plot preset: {preset!r}")
    return _STYLE_PRESETS[preset]


def get_figure_size(kind: str, preset: str = "paper") -> Tuple[float, float]:
    """根据图类型和 preset 获取标准画布尺寸。"""
    style = get_style(preset)
    mapping = {
        "single": style.single_size,
        "wide": style.wide_size,
        "sensor_3row": style.sensor_3row_size,
        "sensor_2row": style.sensor_2row_size,
        "sensor_4x2": style.sensor_4x2_size,
        "compare_3row": style.compare_3row_size,
        "rollout_3row": style.rollout_3row_size,
        "component_3x3": style.component_3x3_size,
    }
    if kind not in mapping:
        raise KeyError(f"Unknown figure kind: {kind!r}")
    return mapping[kind]


def get_xyz_styles() -> Dict[str, SeriesStyle]:
    """返回 X/Y/Z 三轴的稳定视觉编码。"""
    return dict(_XYZ_SERIES)


def get_group_styles() -> Dict[str, SeriesStyle]:
    """返回 Acc/Gyro/Vel 三组的稳定视觉编码。"""
    return dict(_GROUP_SERIES)


def get_observed_pred_styles() -> Dict[str, SeriesStyle]:
    """返回监督目标与预测曲线的默认样式。"""
    return dict(_OBSERVED_PRED_SERIES)


def infer_model_role(label: str, explicit_role: Optional[str] = None) -> str:
    """根据标签文本或显式提示推断模型角色。"""
    if explicit_role is not None:
        role = explicit_role.lower()
        if role not in _MODEL_ROLE_SERIES:
            raise KeyError(f"Unknown model role: {explicit_role!r}")
        return role

    lowered = label.lower()
    if any(tok in lowered for tok in ("ours", "our", "proposed", "primary")):
        return "primary"
    if any(tok in lowered for tok in ("ablation", "abl")):
        return "ablation"
    return "baseline"


def get_model_role_style(role: str) -> SeriesStyle:
    """返回 primary/baseline/ablation 对应的曲线样式。"""
    role_key = role.lower()
    if role_key not in _MODEL_ROLE_SERIES:
        raise KeyError(f"Unknown model role: {role!r}")
    return _MODEL_ROLE_SERIES[role_key]


def setup_mpl(style: Optional[MplGlobalStyle] = None) -> None:
    """把仓库统一科研绘图风格写入 matplotlib 全局配置。"""
    style = style or get_style("paper")
    plt.rcParams.update(
        {
            "font.family": style.font_family,
            "font.serif": list(style.font_serif),
            "font.size": style.base_fontsize,
            "axes.labelsize": style.labelsize,
            "axes.titlesize": style.labelsize,
            "xtick.labelsize": style.ticksize,
            "ytick.labelsize": style.ticksize,
            "legend.fontsize": style.legendsize,
            "axes.grid": style.grid,
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.major.size": style.tick_length,
            "xtick.major.width": style.tick_width,
            "ytick.major.size": style.tick_length,
            "ytick.major.width": style.tick_width,
            "lines.linewidth": style.default_linewidth,
            "legend.frameon": style.legend_frameon,
            "legend.borderaxespad": style.legend_borderaxespad,
            "axes.prop_cycle": cycler(color=list(style.prop_cycle)),
            "figure.figsize": style.single_size,
            "figure.dpi": style.figure_dpi,
            "savefig.dpi": style.savefig_dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def apply_axes_style(ax: plt.Axes, *, grid: Optional[bool] = None, grid_alpha: Optional[float] = None) -> None:
    """对单个坐标轴应用统一边框、刻度和网格风格。"""
    style = get_style("paper")
    use_grid = style.grid if grid is None else grid
    ax.set_facecolor("white")
    ax.tick_params(direction="out")
    if use_grid:
        ax.grid(True, alpha=style.grid_alpha if grid_alpha is None else grid_alpha)
    else:
        ax.grid(False)


def apply_shared_xlabels(axes: Sequence[plt.Axes], xlabel: str) -> None:
    """为共享 x 轴的一组坐标轴设置底部公共 x 标签。"""
    if len(axes) == 0:
        return
    for ax in axes[:-1]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", labelbottom=False)
    axes[-1].set_xlabel(xlabel)


def apply_minimal_legend(legend: Optional[plt.Legend]) -> None:
    """把图例收敛到仓库统一的极简样式。"""
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_alpha(0.0)
    frame.set_linewidth(0.0)
    frame.set_facecolor("white")
    frame.set_edgecolor("none")


def save_figure(fig: plt.Figure, out_stem: Path, fmt: str = "png") -> None:
    """按约定格式导出图片文件。"""
    stem = Path(out_stem)
    if fmt in ("png", "both"):
        fig.savefig(stem.with_suffix(".png"))
    if fmt in ("pdf", "both"):
        fig.savefig(stem.with_suffix(".pdf"))


@dataclass(frozen=True)
class TrajStyle:
    """二维轨迹线与起终点的样式配置。"""
    traj_lw: float = 1.0
    traj_alpha: float = 0.95
    start_marker: str = "o"
    end_marker: str = "o"
    start_s: float = 18.0
    end_s: float = 22.0
    marker_edge_lw: float = 0.9
    traj_color: str = "#4C78A8"
    depth_color: str = "#59A14F"
    start_face: str = "white"
    end_face: Optional[str] = None


_TRAJ = TrajStyle()


def apply_axes_2d(ax: plt.Axes) -> None:
    """对二维轨迹图坐标轴应用统一样式。"""
    apply_axes_style(ax, grid=False)


def plot_start_end(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    ts: TrajStyle = _TRAJ,
    *,
    label_start: Optional[str] = "Start",
    label_end: Optional[str] = "End",
) -> None:
    """在二维轨迹图上标注起点与终点。"""
    ax.scatter(
        [x0],
        [y0],
        s=ts.start_s,
        marker=ts.start_marker,
        facecolors=ts.start_face,
        edgecolors=color,
        linewidths=ts.marker_edge_lw,
        zorder=3,
        label=label_start,
    )

    end_face = ts.end_face if ts.end_face is not None else color
    ax.scatter(
        [x1],
        [y1],
        s=ts.end_s,
        marker=ts.end_marker,
        facecolors=end_face,
        edgecolors=color,
        linewidths=ts.marker_edge_lw,
        zorder=3,
        label=label_end,
    )


def plot_traj_line(
    ax: plt.Axes,
    x,
    y,
    *,
    color: str,
    label: Optional[str] = None,
    ts: TrajStyle = _TRAJ,
) -> None:
    """绘制一条二维轨迹折线。"""
    ax.plot(x, y, color=color, linewidth=ts.traj_lw, alpha=ts.traj_alpha, label=label, zorder=2)


def get_figsize_two_panels(style: Optional[MplGlobalStyle] = None) -> Tuple[float, float]:
    """返回两面板布局常用的标准画布尺寸。"""
    active = style or get_style("paper")
    return active.wide_size
