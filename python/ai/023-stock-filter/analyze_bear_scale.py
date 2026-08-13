#!/usr/bin/env python
"""
analyze_bear_scale.py — 复盘 backtest 输出的 bear_scale.csv

读取 ``<backtest-output>/bear_scale.csv`` (date, scale) 并产出：
  1. 控制台文本报告 — 整体分布、按年分段、最大防御/进攻区间、季度热力图
  2. 可选 PNG — 年度档位堆叠柱状图 + scale 时序曲线 + 档位色块

典型用法::

    ~/.pyenv/versions/qlib/bin/python analyze_bear_scale.py \\
        --csv .cache/backtest/bear_scale.csv \\
        --plot

每次回测后跑一遍，新数据会自动覆盖旧报告。详见 ``docs/analyze-bear-scale.md``。

依赖：pandas / numpy / matplotlib（项目已有的 venv 全覆盖）。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 中文字体回退（按全局 CLAUDE.md 偏好）
# ----------------------------------------------------------------------------


def setup_cjk_font() -> None:
    """配置 matplotlib 中文字体回退：PingFang SC → Hiragino Sans GB → Heiti TC。

    必须在 import pyplot 后立即调用一次，PNG 的中文标题/轴标才不会出 □□。
    """
    import matplotlib
    from matplotlib import font_manager

    preferred = ["PingFang SC", "Hiragino Sans GB", "Heiti TC"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [f for f in preferred if f in available]
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """分析结果聚合，文本报告与出图共用同一份结构化数据。"""

    n_days: int
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    scale_counts: pd.Series  # 各档位天数
    scale_pct: pd.Series  # 各档位占比
    defensive_days: int  # scale < 1.0 的天数
    defensive_pct: float  # 防御占比
    avg_exposure: float  # 加权平均权益暴露
    by_year: pd.DataFrame  # 按年统计
    by_year_quarter: pd.DataFrame  # 按年-季度统计
    max_defense_run: pd.DataFrame  # 最长连续 max-defense 区间
    bull_run: pd.DataFrame  # 最长连续满仓区间
    transitions: pd.DataFrame  # scale 档位跳变事件


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="复盘 backtest 的 bear_scale.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--csv",
        default=".cache/backtest/bear_scale.csv",
        help="bear_scale.csv 路径（默认 .cache/backtest/bear_scale.csv）",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="额外输出 PNG 图（年度堆叠 + 时序曲线）",
    )
    p.add_argument(
        "--output-dir",
        default=".cache/backtest/analysis/",
        help="报告 + PNG 输出目录（默认 .cache/backtest/analysis/）",
    )
    return p.parse_args()


# ----------------------------------------------------------------------------
# 加载与基础校验
# ----------------------------------------------------------------------------


def load_bear_scale(csv_path: Path) -> pd.DataFrame:
    """读取 CSV，校验列结构，返回按日期排序的 DataFrame。

    期望列：``date, scale``。scale 取值集合 = {0.3, 0.4, 0.6, 1.0}。
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    missing = {"date", "scale"} - set(df.columns)
    if missing:
        sys.exit(f"[FATAL] {csv_path} 缺少必要列 {missing}；现有列：{list(df.columns)}")
    df = df.sort_values("date").reset_index(drop=True)
    df["scale"] = df["scale"].astype(float)
    if df["scale"].isna().any():
        n = df["scale"].isna().sum()
        sys.exit(f"[FATAL] scale 列含 {n} 个 NaN，文件可能未完整生成")
    valid_scales = {0.3, 0.4, 0.6, 1.0}
    actual = set(df["scale"].unique())
    unexpected = actual - valid_scales
    if unexpected:
        print(f"[WARN] scale 列出现了预期外的取值 {unexpected}（正常应为 {valid_scales}）")
    return df


# ----------------------------------------------------------------------------
# 分析函数
# ----------------------------------------------------------------------------


def compute_distribution(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """返回 (档位天数, 档位占比)，按 1.0 → 0.6 → 0.4 → 0.3 顺序。"""
    order = [1.0, 0.6, 0.4, 0.3]
    counts = df["scale"].value_counts().reindex(order, fill_value=0).astype(int)
    pct = (counts / counts.sum() * 100).round(2)
    return counts, pct


def weighted_exposure(series: pd.Series) -> float:
    """加权平均权益暴露 = sum(scale_i) / N。"""
    return float(series.mean())


def by_year_table(df: pd.DataFrame) -> pd.DataFrame:
    """按年统计：天数、防御天数、防御占比、平均 scale、加权暴露。

    使用日历年（YYYY）分组，不按回测起点对齐——更贴近 A 股季节性叙事。
    """
    out = df.assign(year=df["date"].dt.year)
    grp = out.groupby("year")
    table = pd.DataFrame(
        {
            "总交易日": grp.size(),
            "防御天数(scale<1)": grp["scale"].apply(lambda s: (s < 1.0).sum()),
            "防御占比%": (grp["scale"].apply(lambda s: (s < 1.0).mean()) * 100).round(1),
            "平均scale": grp["scale"].mean().round(3),
            "max-def天数(scale=0.3)": grp["scale"].apply(lambda s: (s == 0.3).sum()),
        }
    ).astype({"防御天数(scale<1)": int, "max-def天数(scale=0.3)": int})
    return table


def by_year_quarter_table(df: pd.DataFrame) -> pd.DataFrame:
    """按年-季度统计防御占比。"""
    out = df.assign(
        year=df["date"].dt.year,
        quarter=df["date"].dt.quarter,
    )
    grp = out.groupby(["year", "quarter"])
    table = pd.DataFrame(
        {
            "总交易日": grp.size(),
            "防御天数": grp["scale"].apply(lambda s: (s < 1.0).sum()),
            "防御占比%": (grp["scale"].apply(lambda s: (s < 1.0).mean()) * 100).round(1),
            "平均scale": grp["scale"].mean().round(3),
        }
    ).astype({"防御天数": int})
    return table


def longest_run(df: pd.DataFrame, target_scale: float) -> pd.DataFrame:
    """找连续 target_scale 的最长区间，返回按长度降序的 (start, end, length_days)。

    实现：直接用 mask 的差分找 True 区间的起止索引——避免之前用 True 子集索引导致
    错位映射到原 df 的 bug。
    """
    mask = (df["scale"] == target_scale).to_numpy()
    n = len(mask)
    if not mask.any():
        return pd.DataFrame(columns=["start", "end", "length_days"])

    # False→True 的跳变点 +1 是 True 段起点；True→False 是终点
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1  # False→True 跨过的那行就是 True 起点
    ends = np.where(diff == -1)[0]  # True→False 的前一行是 True 终点
    # 处理首尾是 True 的情况
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [n - 1]))

    runs = []
    for s, e in zip(starts, ends):
        runs.append(
            {
                "start": df["date"].iloc[s].date(),
                "end": df["date"].iloc[e].date(),
                "length_days": int(e - s + 1),
            }
        )
    out = (
        pd.DataFrame(runs)
        .sort_values("length_days", ascending=False)
        .reset_index(drop=True)
    )
    return out


def find_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """scale 档位跳变事件：date, prev_scale, new_scale, direction('attack'|'defense')。

    用于回答"哪一天突然从满仓切到防御"或反向。
    """
    shifted = df["scale"].shift(1)
    changed = df["scale"] != shifted
    events = df.loc[changed, ["date", "scale"]].copy()
    events.columns = ["date", "new_scale"]
    events["prev_scale"] = shifted[changed].values
    # direction: new_scale > prev_scale → attack（加回进攻）；反之 defense
    events["direction"] = np.where(
        events["new_scale"] > events["prev_scale"], "attack", "defense"
    )
    return events.reset_index(drop=True)


def run_analysis(df: pd.DataFrame) -> AnalysisResult:
    counts, pct = compute_distribution(df)
    defensive_days = int((df["scale"] < 1.0).sum())
    n = len(df)
    avg_exp = weighted_exposure(df["scale"])
    return AnalysisResult(
        n_days=n,
        date_start=df["date"].min(),
        date_end=df["date"].max(),
        scale_counts=counts,
        scale_pct=pct,
        defensive_days=defensive_days,
        defensive_pct=defensive_days / n * 100,
        avg_exposure=avg_exp,
        by_year=by_year_table(df),
        by_year_quarter=by_year_quarter_table(df),
        max_defense_run=longest_run(df, 0.3),
        bull_run=longest_run(df, 1.0),
        transitions=find_transitions(df),
    )


# ----------------------------------------------------------------------------
# 文本报告
# ----------------------------------------------------------------------------


def render_text_report(result: AnalysisResult) -> str:
    """拼接控制台可读文本报告。"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(" bear_scale.csv 复盘报告")
    lines.append("=" * 70)
    lines.append(
        f"区间: {result.date_start.date()} → {result.date_end.date()}   "
        f"总交易日: {result.n_days}"
    )
    lines.append("")
    lines.append("--- 整体分布 ---")
    for scale in result.scale_counts.index:
        cnt = result.scale_counts[scale]
        p = result.scale_pct[scale]
        tag = (
            "满仓"
            if scale == 1.0
            else (
                "1 触发"
                if scale == 0.6
                else ("2 触发" if scale == 0.4 else "3 触发(max-def)")
            )
        )
        lines.append(f"  scale={scale:<4} ({tag:<16}): {cnt:>4} 天  {p:>5.2f}%")
    lines.append("")
    lines.append(
        f"  防御天数 (scale<1): {result.defensive_days}  占比 {result.defensive_pct:.1f}%"
    )
    lines.append(f"  加权平均权益暴露: {result.avg_exposure:.3f}")
    lines.append("")
    lines.append("--- 按年画像 ---")
    lines.append(result.by_year.to_string())
    lines.append("")
    lines.append("--- 按年-季度 (防御占比%) ---")
    lines.append(result.by_year_quarter.to_string())
    lines.append("")
    lines.append("--- 最长连续 max-defense (scale=0.3) 区间 ---")
    if len(result.max_defense_run) == 0:
        lines.append("  无")
    else:
        head = result.max_defense_run.head(5)
        lines.append(head.to_string(index=False))
    lines.append("")
    lines.append("--- 最长连续满仓 (scale=1.0) 区间 ---")
    if len(result.bull_run) == 0:
        lines.append("  无")
    else:
        head = result.bull_run.head(5)
        lines.append(head.to_string(index=False))
    lines.append("")
    lines.append(
        f"--- scale 档位跳变事件（总 {len(result.transitions)} 次）---"
    )
    if len(result.transitions):
        head = result.transitions.head(15)
        lines.append(head.to_string(index=False))
        if len(result.transitions) > 15:
            lines.append(f"  ... (其余 {len(result.transitions) - 15} 次省略)")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------------


def plot_yearly_breakdown(result: AnalysisResult, save_path: Path) -> None:
    """年度档位堆叠柱状图：每年 1.0/0.6/0.4/0.3 各占多少天。"""
    import matplotlib.pyplot as plt

    by_year = result.by_year.copy()
    # 算出每年各档位天数
    # 注意：result.by_year 不含分档统计，需要从原 df 重算
    # 简单做法：从 transitions 信息里无法还原，所以这里重新跑一次
    # —— 不在 AnalysisResult 里塞这些字段以保持轻量；改为在调用前传入 df。
    raise NotImplementedError("plot_yearly_breakdown 需要原始 df，调用方请改用 plot_yearly_breakdown_v2")


def plot_yearly_breakdown_v2(df: pd.DataFrame, result: AnalysisResult, save_path: Path) -> None:
    """年度档位堆叠柱状图：每年 1.0/0.6/0.4/0.3 各占多少天。"""
    import matplotlib.pyplot as plt

    years = sorted(df["date"].dt.year.unique())
    matrix = np.zeros((len(years), 4))  # 1.0/0.6/0.4/0.3
    for i, y in enumerate(years):
        sub = df[df["date"].dt.year == y]["scale"]
        for j, s in enumerate([1.0, 0.6, 0.4, 0.3]):
            matrix[i, j] = (sub == s).sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c", "#ffbb78", "#ff7f0e", "#d62728"]
    labels = ["1.0 满仓", "0.6 1触发", "0.4 2触发", "0.3 max-def"]
    bottom = np.zeros(len(years))
    for j in range(4):
        ax.bar(years, matrix[:, j], bottom=bottom, color=colors[j], label=labels[j])
        bottom += matrix[:, j]
    ax.set_title("bear_scale 各档位年度分布")
    ax.set_xlabel("年份")
    ax.set_ylabel("天数")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_timeline(df: pd.DataFrame, save_path: Path) -> None:
    """scale 时序曲线 + 档位色块背景。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.5))
    # 档位色块背景
    df_sorted = df.sort_values("date").reset_index(drop=True)
    palette = {1.0: "#e6f4e6", 0.6: "#fff2cc", 0.4: "#ffd6a5", 0.3: "#ffb3b3"}
    for scale, color in palette.items():
        mask = df_sorted["scale"] == scale
        if not mask.any():
            continue
        # 用 fill_between 给整段时间上色
        runs = (mask != mask.shift()).cumsum()
        for rid in runs[mask].unique():
            seg = df_sorted[mask & (runs == rid)]
            if len(seg) > 1:
                ax.axvspan(
                    seg["date"].iloc[0],
                    seg["date"].iloc[-1],
                    color=color,
                    alpha=0.5,
                    zorder=0,
                )
    ax.plot(df_sorted["date"], df_sorted["scale"], color="#333333", linewidth=1.2, zorder=2)
    ax.set_yticks([0.3, 0.4, 0.6, 1.0])
    ax.set_yticklabels(["0.3", "0.4", "0.6", "1.0"])
    ax.set_title("bear_scale 时序（色块：满仓绿/1触发黄/2触发橙/max-def红）")
    ax.set_ylabel("scale")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"[FATAL] bear_scale.csv 找不到：{csv_path}")

    df = load_bear_scale(csv_path)
    result = run_analysis(df)

    report = render_text_report(result)
    print(report)

    if args.plot or args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "bear_scale_report.txt"
        report_path.write_text(report, encoding="utf-8")
        print(f"[OK] 文本报告 → {report_path}")

        if args.plot:
            try:
                import matplotlib.pyplot as plt  # noqa: F401  确认依赖可用
            except ImportError:
                sys.exit("[FATAL] --plot 需要 matplotlib")
            setup_cjk_font()  # 必须先设字体再画图
            plot_yearly_breakdown_v2(
                df, result, out_dir / "bear_scale_yearly.png"
            )
            plot_timeline(df, out_dir / "bear_scale_timeline.png")
            print(f"[OK] PNG → {out_dir/'bear_scale_yearly.png'}")
            print(f"[OK] PNG → {out_dir/'bear_scale_timeline.png'}")


if __name__ == "__main__":
    main()