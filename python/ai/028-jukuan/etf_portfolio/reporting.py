"""报告生成工具。

主要内容：
    - nav_curve: 计算组合净值曲线
    - plot_nav_compare: 多组合净值曲线对比图
    - plot_decay_distribution: 衰减率分布直方图
    - plot_robustness_heatmap: 稳健度热图
    - write_report_html: 一键输出 HTML 报告（可选）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def nav_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    """日收益 → 累计净值。"""
    return start * (1 + returns).cumprod()


def plot_nav_compare(
    port_returns: dict[str, pd.Series],
    title: str = "组合净值曲线对比",
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (12, 6),
    log_scale: bool = False,
) -> None:
    """多条组合净值曲线对比图。

    Args:
        port_returns: {name: returns_series}
        save_path: 如指定则保存为 PNG
        log_scale: 是否对数纵轴
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, ret in port_returns.items():
        if ret is None or ret.empty:
            continue
        nav = nav_curve(ret)
        ax.plot(nav.index, nav.values, label=name, linewidth=1.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("累计净值")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale("log")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_decay_distribution(
    wfa_metrics: pd.DataFrame,
    save_path: Path | str | None = None,
) -> None:
    """衰减率分布直方图（按目标分组）。"""
    if "objective" not in wfa_metrics.columns:
        wfa_metrics = wfa_metrics.copy()
        wfa_metrics["objective"] = "default"
    objectives = sorted(wfa_metrics["objective"].unique())
    fig, axes = plt.subplots(1, len(objectives), figsize=(5 * len(objectives), 4), sharey=True)
    if len(objectives) == 1:
        axes = [axes]
    for ax, obj in zip(axes, objectives):
        sub = wfa_metrics[wfa_metrics["objective"] == obj]["decay"].dropna()
        if sub.empty:
            ax.set_title(f"{obj} (无数据)")
            continue
        ax.hist(sub, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
        for x, color, label in [(0.30, "green", "优秀 0.30"), (0.60, "orange", "尚可 0.60"), (0.70, "red", "过拟合 0.70")]:
            ax.axvline(x, color=color, linestyle="--", linewidth=1, label=label)
        ax.set_title(obj)
        ax.set_xlabel("衰减率")
        ax.set_ylabel("频次")
        ax.legend(fontsize=8)
    fig.suptitle("WFA 衰减率分布", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_robustness_heatmap(
    score_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Path | str | None = None,
) -> None:
    """稳健度评分热图（Top N ETF × 指标）。"""
    sub = score_df.head(top_n)[["freq", "avg_weight", "stability", "n_obj"]]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(sub))))
    # 归一化到 [0, 1] 用于热图
    norm = (sub - sub.min()) / (sub.max() - sub.min() + 1e-9)
    im = ax.imshow(norm.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(sub.columns)))
    ax.set_xticklabels(sub.columns, rotation=0)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub.index, fontsize=9)
    # 在格子里写值
    for i in range(len(sub)):
        for j in range(len(sub.columns)):
            ax.text(j, i, f"{sub.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="归一化值")
    ax.set_title(f"ETF 稳健度评分 Top {top_n}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def summarize_baselines(
    port_returns: dict[str, pd.Series],
    rf: float = 0.025,
) -> pd.DataFrame:
    """对组合字典统一算指标。"""
    from .metrics import full_metrics
    rows = []
    for name, ret in port_returns.items():
        m = full_metrics(ret, rf=rf)
        rows.append({
            "组合":       name,
            "年化收益":   m.get("annual_return"),
            "年化波动":   m.get("annual_vol"),
            "夏普":       m.get("sharpe"),
            "卡玛":       m.get("calmar"),
            "最大回撤":   m.get("max_drawdown"),
        })
    df = pd.DataFrame(rows).sort_values("夏普", ascending=False).reset_index(drop=True)
    return df
