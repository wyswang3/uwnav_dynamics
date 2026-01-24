# src/uwnav_dynamics/viz/style/imu_style.py
from __future__ import annotations

"""
IMU plotting style policy (layout + ticks + legend), shared by:
  - raw IMU plots
  - frame-transformed IMU plots
  - filtered / bias-corrected IMU plots

Design goals:
  1) Anchored composition: fixed canvas + fixed margins + stable spacing.
  2) Typography scales with canvas (so figures remain readable in papers).
  3) Tick policy:
       - x: stable density (MaxNLocator nbins)
       - y: prefer 3 "nice" ticks; fallback to 25/50/75 positions if needed
  4) Legend policy: semi-transparent white background.
  5) Axis color semantics: X=C0, Y=C1, Z=C2 (consistent across all IMU figures)

This module is style-only. No data IO, no filtering, no alignment.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -----------------------------
# Layout & visual policy
# -----------------------------
@dataclass(frozen=True)
class Imu3RowLayout:
    # canvas
    fig_w_in: float = 5.0
    fig_h_in: float = 4.8
    dpi: int = 450

    # margins
    left: float = 0.12
    right: float = 0.98
    bottom: float = 0.14
    top: float = 0.93

    # spacing
    hspace: float = 0.26

    # tick policy
    x_nbins: int = 6
    y_nticks: int = 3
    y_pad_frac: float = 0.03

    # legend policy
    legend_alpha: float = 0.75
    legend_face_rgba: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.75)

    # typography scaling reference
    ref_in: float = 5.0

    # ---- derived ----
    def scale(self) -> float:
        return min(self.fig_w_in, self.fig_h_in) / self.ref_in

    def title_fs(self) -> float:
        return 10.0 * self.scale()

    def label_fs(self) -> float:
        return 9.0 * self.scale()

    def tick_fs(self) -> float:
        return 8.0 * self.scale()

    def legend_fs(self) -> float:
        return 8.0 * self.scale()

    def lw(self) -> float:
        return 1.1 * self.scale()

    def tick_len_major(self) -> float:
        return 3.2 * self.scale()

    def tick_len_minor(self) -> float:
        return 2.0 * self.scale()

    def tick_w_major(self) -> float:
        return 0.8 * self.scale()

    def tick_w_minor(self) -> float:
        return 0.6 * self.scale()


# Default color semantics for axes
IMU_AXIS_COLORS: Tuple[str, str, str] = ("C0", "C1", "C2")
IMU_AXIS_LABELS: Tuple[str, str, str] = ("X axis", "Y axis", "Z axis")


# -----------------------------
# Legend styling
# -----------------------------
def apply_legend_style(
    leg: Optional[plt.Legend],
    *,
    alpha: float,
    face_rgba: Tuple[float, float, float, float],
) -> None:
    if leg is None:
        return
    fr = leg.get_frame()
    fr.set_alpha(alpha)
    fr.set_facecolor(face_rgba)
    fr.set_edgecolor("none")


# -----------------------------
# Tick policy: y ticks (prefer 3 "nice" ticks)
# -----------------------------
def set_y_ticks_pretty_3(ax: plt.Axes, *, y_pad_frac: float = 0.03) -> None:
    """
    Prefer "nice" ticks close to 3 ticks.
    Fallback to 25/50/75 axis positions if locator degenerates.

    Outcome: typically 3 major ticks, visually pleasing and robust.
    """
    ax.relim()
    ax.autoscale(enable=True, axis="y", tight=False)

    y0, y1 = ax.get_ylim()
    if not (np.isfinite(y0) and np.isfinite(y1)):
        return

    span = float(y1 - y0)
    if span == 0.0:
        eps = 1e-6 if y0 == 0.0 else abs(y0) * 1e-3
        y0, y1 = y0 - eps, y1 + eps
        span = float(y1 - y0)

    # padding
    pad = abs(span) * float(y_pad_frac)
    ax.set_ylim(y0 - pad, y1 + pad)
    y0, y1 = ax.get_ylim()

    locator = mticker.MaxNLocator(nbins=3, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10])
    ax.yaxis.set_major_locator(locator)

    ticks = ax.get_yticks()
    ticks = ticks[np.isfinite(ticks)]
    ticks_in = ticks[(ticks >= y0 - 1e-12) & (ticks <= y1 + 1e-12)]

    if len(ticks_in) < 3:
        pos = np.array([0.25, 0.5, 0.75], dtype=float)
        ax.set_yticks(y0 + (y1 - y0) * pos)
        return

    if len(ticks_in) > 3:
        mid = 0.5 * (y0 + y1)
        order = np.argsort(np.abs(ticks_in - mid))
        sel = np.sort(ticks_in[order[:3]])
        ax.set_yticks(sel)
        return

    ax.set_yticks(ticks_in[:3])


# -----------------------------
# Canvas creation helpers
# -----------------------------
def make_imu_3rows_canvas(
    layout: Imu3RowLayout = Imu3RowLayout(),
) -> Tuple[plt.Figure, Sequence[plt.Axes], Imu3RowLayout]:
    """
    Create a 3x1 shared-x canvas with fixed margins/spacings and tick policy applied.
    The caller only needs to plot data and set titles/labels.
    """
    fig, axes = plt.subplots(
        3, 1, sharex=True,
        figsize=(layout.fig_w_in, layout.fig_h_in),
        dpi=layout.dpi,
        gridspec_kw={"hspace": layout.hspace},
    )

    x_locator = mticker.MaxNLocator(nbins=layout.x_nbins)

    for i, ax in enumerate(axes):
        ax.xaxis.set_major_locator(x_locator)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=layout.tick_fs(),
            width=layout.tick_w_major(),
            length=layout.tick_len_major(),
            direction="out",
        )
        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=layout.tick_fs(),
            width=layout.tick_w_minor(),
            length=layout.tick_len_minor(),
            direction="out",
        )
        if i < 2:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    # anchored margins
    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
    )

    return fig, axes, layout


def finalize_imu_axes(
    axes: Sequence[plt.Axes],
    *,
    y_pad_frac: float = 0.03,
) -> None:
    """
    Apply y-tick policy to all axes.
    Call after plotting data on each axis.
    """
    for ax in axes:
        set_y_ticks_pretty_3(ax, y_pad_frac=y_pad_frac)


def plot_xyz_lines(
    ax: plt.Axes,
    t_s: np.ndarray,
    xyz: np.ndarray,
    *,
    linewidth: float,
    colors: Tuple[str, str, str] = IMU_AXIS_COLORS,
) -> Tuple[plt.Line2D, plt.Line2D, plt.Line2D]:
    """
    Convenience helper: plot 3-axis curves with fixed color semantics.
    xyz must be (N,3).
    """
    x = np.asarray(xyz, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {x.shape}")
    l1, = ax.plot(t_s, x[:, 0], linewidth=linewidth, color=colors[0])
    l2, = ax.plot(t_s, x[:, 1], linewidth=linewidth, color=colors[1])
    l3, = ax.plot(t_s, x[:, 2], linewidth=linewidth, color=colors[2])
    return l1, l2, l3


def add_xyz_legend(
    ax: plt.Axes,
    lines: Tuple[plt.Line2D, plt.Line2D, plt.Line2D],
    layout: Imu3RowLayout,
    *,
    labels: Tuple[str, str, str] = IMU_AXIS_LABELS,
    loc: str = "upper right",
) -> None:
    leg = ax.legend(
        list(lines),
        list(labels),
        loc=loc,
        frameon=True,
        fontsize=layout.legend_fs(),
    )
    apply_legend_style(leg, alpha=layout.legend_alpha, face_rgba=layout.legend_face_rgba)
