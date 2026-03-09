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
- 多曲线比较图应优先使用高对比、近互补的配色组合，让 primary / baseline / ablation 或多条候选曲线在首眼观察时就能分离。
- 正式 preset 采用纯白底、无背景网格的科研风格，避免图面噪声抢占曲线注意力。
- 英文、数字与 mathtext 统一收敛到 `Times New Roman`，文字颜色统一使用纯黑。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
from cycler import cycler


_FIGURE_BG = "#FFFFFF"
_AXES_BG = "#FFFFFF"
_TIMES_NEW_ROMAN = "Times New Roman"
_TEXT_PRIMARY = "#000000"
_TEXT_SECONDARY = "#000000"
_GRID_COLOR = "#D9E2EC"
_SPINE_COLOR = "#C7D0D9"

_OBSERVED_COLOR = "#2F3B52"
_PRED_COLOR = "#2C7FB8"
_PRED_ALT_COLOR = "#E76F51"
_BASELINE_COLOR = "#5C6BC0"
_PRIMARY_COLOR = "#1B9E77"
_ABLATION_COLOR = "#E76F51"
_UNCERTAINTY_FILL = "#A9D6E5"
_RESIDUAL_COLOR = "#C8553D"
_MUTED_COLOR = "#B8C4CF"
_ACCENT_GOLD = "#D4A72C"
_ACCENT_ROSE = "#D65A7A"
_TIMES_FONT_FILES: Tuple[str, ...] = (
    "Times_New_Roman.ttf",
    "Times_New_Roman_Italic.ttf",
    "Times_New_Roman_Bold.ttf",
    "Times_New_Roman_Bold_Italic.ttf",
    "times.ttf",
    "timesi.ttf",
    "timesbd.ttf",
    "timesbi.ttf",
)


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
    legend_handlelength: float
    legend_columnspacing: float


_STYLE_PRESETS: Dict[str, MplGlobalStyle] = {
    "paper": MplGlobalStyle(
        preset="paper",
        font_family=_TIMES_NEW_ROMAN,
        font_serif=(_TIMES_NEW_ROMAN, "Times", "DejaVu Serif"),
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
            _PRED_COLOR,
            "#F28E2B",
            _PRIMARY_COLOR,
            _PRED_ALT_COLOR,
            _BASELINE_COLOR,
            _ACCENT_ROSE,
        ),
        legend_frameon=False,
        legend_borderaxespad=0.25,
        legend_handlelength=2.4,
        legend_columnspacing=1.1,
    ),
    "ppt": MplGlobalStyle(
        preset="ppt",
        font_family=_TIMES_NEW_ROMAN,
        font_serif=(_TIMES_NEW_ROMAN, "Times", "DejaVu Serif"),
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
            _PRED_COLOR,
            "#F28E2B",
            _PRIMARY_COLOR,
            _PRED_ALT_COLOR,
            _BASELINE_COLOR,
            _ACCENT_ROSE,
        ),
        legend_frameon=False,
        legend_borderaxespad=0.25,
        legend_handlelength=2.4,
        legend_columnspacing=1.2,
    ),
}


_XYZ_SERIES: Dict[str, SeriesStyle] = {
    "x": SeriesStyle(color=_PRED_COLOR),
    "y": SeriesStyle(color="#F28E2B"),
    "z": SeriesStyle(color=_PRIMARY_COLOR),
}

_GROUP_SERIES: Dict[str, SeriesStyle] = {
    "Acc": SeriesStyle(color=_PRED_COLOR),
    "Gyro": SeriesStyle(color=_PRED_ALT_COLOR),
    "Vel": SeriesStyle(color=_PRIMARY_COLOR),
}

_OBSERVED_PRED_SERIES: Dict[str, SeriesStyle] = {
    "observed": SeriesStyle(color=_OBSERVED_COLOR, linestyle="-", linewidth=1.7, alpha=0.96, zorder=5),
    "pred": SeriesStyle(color=_PRED_COLOR, linestyle="--", linewidth=1.55, alpha=0.98, zorder=4),
}

_MODEL_ROLE_SERIES: Dict[str, SeriesStyle] = {
    "primary": SeriesStyle(color=_PRIMARY_COLOR, linestyle="-", linewidth=1.95, alpha=1.0, zorder=5),
    "baseline": SeriesStyle(color=_BASELINE_COLOR, linestyle="--", linewidth=1.45, alpha=0.94, zorder=3),
    "ablation": SeriesStyle(color=_ABLATION_COLOR, linestyle="-.", linewidth=1.6, alpha=0.97, zorder=4),
}

_MODEL_ROLE_PALETTES: Dict[str, Tuple[str, ...]] = {
    "primary": (
        _PRIMARY_COLOR,
        _PRED_COLOR,
        _ACCENT_ROSE,
        "#4C956C",
    ),
    "baseline": (
        _BASELINE_COLOR,
        "#7A8CA3",
        "#8FA1B3",
    ),
    "ablation": (
        _ABLATION_COLOR,
        "#F28E2B",
        _RESIDUAL_COLOR,
        _ACCENT_GOLD,
    ),
}


def get_style(preset: str = "paper") -> MplGlobalStyle:
    """按 preset 名称返回全局绘图风格对象。"""
    if preset not in _STYLE_PRESETS:
        raise KeyError(f"Unknown plot preset: {preset!r}")
    return _STYLE_PRESETS[preset]


def _register_times_new_roman() -> None:
    """向 matplotlib 显式注册 Times New Roman，避免字体缓存回退到默认字体。"""
    try:
        font_manager.findfont(_TIMES_NEW_ROMAN, fallback_to_default=False)
        return
    except ValueError:
        pass

    for path_str in font_manager.findSystemFonts():
        if Path(path_str).name in _TIMES_FONT_FILES:
            font_manager.fontManager.addfont(path_str)


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


def get_model_role_styles(roles: Sequence[str]) -> List[SeriesStyle]:
    """
    为一组模型角色生成稳定且可区分的曲线样式。

    规则：
    - 角色决定线型、线宽和视觉层级；
    - 同角色内用 role-specific palette 轮换颜色，避免多条 primary/baseline 曲线难以区分。
    """
    counts: Dict[str, int] = {key: 0 for key in _MODEL_ROLE_SERIES}
    styles: List[SeriesStyle] = []
    for role in roles:
        role_key = role.lower()
        if role_key not in _MODEL_ROLE_SERIES:
            raise KeyError(f"Unknown model role: {role!r}")
        base = _MODEL_ROLE_SERIES[role_key]
        palette = _MODEL_ROLE_PALETTES[role_key]
        color = palette[counts[role_key] % len(palette)]
        counts[role_key] += 1
        styles.append(
            SeriesStyle(
                color=color,
                linestyle=base.linestyle,
                linewidth=base.linewidth,
                alpha=base.alpha,
                zorder=base.zorder,
            )
        )
    return styles


def setup_mpl(style: Optional[MplGlobalStyle] = None) -> None:
    """把仓库统一科研绘图风格写入 matplotlib 全局配置。"""
    style = style or get_style("paper")
    _register_times_new_roman()
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
            "figure.facecolor": _FIGURE_BG,
            "figure.edgecolor": _FIGURE_BG,
            "axes.facecolor": _AXES_BG,
            "axes.edgecolor": _SPINE_COLOR,
            "axes.labelcolor": _TEXT_PRIMARY,
            "axes.titlecolor": _TEXT_PRIMARY,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "text.color": _TEXT_PRIMARY,
            "xtick.color": _TEXT_SECONDARY,
            "ytick.color": _TEXT_SECONDARY,
            "xtick.major.size": style.tick_length,
            "xtick.major.width": style.tick_width,
            "ytick.major.size": style.tick_length,
            "ytick.major.width": style.tick_width,
            "lines.linewidth": style.default_linewidth,
            "legend.frameon": style.legend_frameon,
            "legend.borderaxespad": style.legend_borderaxespad,
            "legend.facecolor": _FIGURE_BG,
            "legend.edgecolor": "none",
            "axes.prop_cycle": cycler(color=list(style.prop_cycle)),
            "grid.color": _GRID_COLOR,
            "mathtext.fontset": "custom",
            "mathtext.rm": _TIMES_NEW_ROMAN,
            "mathtext.it": f"{_TIMES_NEW_ROMAN}:italic",
            "mathtext.bf": f"{_TIMES_NEW_ROMAN}:bold",
            "figure.figsize": style.single_size,
            "figure.dpi": style.figure_dpi,
            "savefig.dpi": style.savefig_dpi,
            "savefig.bbox": "tight",
            "savefig.facecolor": _FIGURE_BG,
            "savefig.pad_inches": 0.02,
        }
    )


def apply_axes_style(ax: plt.Axes, *, grid: Optional[bool] = None, grid_alpha: Optional[float] = None) -> None:
    """对单个坐标轴应用统一边框、刻度和网格风格。"""
    style = get_style("paper")
    use_grid = style.grid if grid is None else grid
    ax.set_facecolor(_AXES_BG)
    for spine in ax.spines.values():
        spine.set_color(_SPINE_COLOR)
        spine.set_linewidth(style.axes_linewidth)
    ax.tick_params(direction="out", colors=_TEXT_SECONDARY)
    ax.xaxis.label.set_color(_TEXT_PRIMARY)
    ax.yaxis.label.set_color(_TEXT_PRIMARY)
    ax.title.set_color(_TEXT_PRIMARY)
    if use_grid:
        ax.grid(True, color=_GRID_COLOR, alpha=style.grid_alpha if grid_alpha is None else grid_alpha)
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


def align_ylabels(axes: Sequence[plt.Axes]) -> None:
    """把同一列多子图的 y 轴标签对齐到统一竖线。"""
    if len(axes) == 0:
        return
    fig = axes[0].figure
    fig.canvas.draw()
    fig.align_ylabels(list(axes))


def apply_minimal_legend(legend: Optional[plt.Legend]) -> None:
    """把图例收敛到仓库统一的极简样式。"""
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_alpha(0.0)
    frame.set_linewidth(0.0)
    frame.set_facecolor("none")
    frame.set_edgecolor("none")
    for text in legend.get_texts():
        text.set_color(_TEXT_PRIMARY)
    legend.set_title(None)


def add_figure_legend(
    fig: plt.Figure,
    handles,
    labels,
    *,
    ncol: Optional[int] = None,
    y: float = 1.01,
    loc: str = "lower center",
) -> Optional[plt.Legend]:
    """在画布顶部空白区放置统一图例，避免遮挡坐标轴内部数据。"""
    labels_list = list(labels)
    handles_list = list(handles)
    if len(handles_list) == 0 or len(labels_list) == 0:
        return None

    style = get_style("paper")
    legend = fig.legend(
        handles_list,
        labels_list,
        loc=loc,
        bbox_to_anchor=(0.5, y),
        ncol=min(len(labels_list), 4) if ncol is None else int(ncol),
        frameon=True,
        handlelength=style.legend_handlelength,
        columnspacing=style.legend_columnspacing,
        borderaxespad=0.0,
    )
    apply_minimal_legend(legend)
    return legend


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
