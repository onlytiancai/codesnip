# 分析 backtest 输出的 bear_scale.csv — 使用文档

## 概述

`analyze_bear_scale.py` 复盘 `backtest_etf_rotation.py` 的产出 `bear_scale.csv`，
给出按年/季度的防御比例、连续 max-defense / 满仓区间、scale 跳变事件，
以及 2 张 PNG（年度堆叠柱状图 + 时序曲线+档位色块）。

**典型场景**：每次跑完回测拿到新的 `.cache/backtest/bear_scale.csv`，
跑一遍本脚本得到量化数据 + 图表，再让 AI 结合 `metrics.txt` 写解读。

## 安装依赖

项目 venv (`~/.pyenv/versions/qlib/bin/python`) 已覆盖：pandas / numpy / matplotlib。
**无需额外安装**。

## 运行

```bash
~/.pyenv/versions/qlib/bin/python analyze_bear_scale.py \
  --csv .cache/backtest/bear_scale.csv \
  --plot \
  --output-dir .cache/backtest/analysis/
```

### CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--csv` | `.cache/backtest/bear_scale.csv` | bear_scale.csv 路径 |
| `--plot` | 关 | 是否额外输出 PNG |
| `--output-dir` | `.cache/backtest/analysis/` | 报告 + PNG 输出目录 |

## 输出文件

```
.cache/backtest/analysis/
├── bear_scale_report.txt       # 文本报告（控制台也会 print 一份）
├── bear_scale_yearly.png       # 年度档位堆叠柱状图
└── bear_scale_timeline.png     # scale 时序曲线 + 档位色块
```

### 文本报告包含的字段

```
======================================================================
 bear_scale.csv 复盘报告
======================================================================
区间: 2021-04-01 → 2026-08-12   总交易日: 1301

--- 整体分布 ---
  scale=1.0  (满仓              ):  488 天  37.51%
  scale=0.6  (1 触发            ):  299 天  22.98%
  scale=0.4  (2 触发            ):  473 天  36.36%
  scale=0.3  (3 触发(max-def)   ):   41 天   3.15%
  防御天数 (scale<1): 813  占比 62.5%
  加权平均权益暴露: 0.668

--- 按年画像 ---           # 防御天数 / 占比 / 平均 scale / max-def 天数
--- 按年-季度 ---          # 季度粒度的防御占比
--- 最长连续 max-defense ---  # scale=0.3 区间
--- 最长连续满仓 ---       # scale=1.0 区间
--- scale 跳变事件 ---      # 每次档位切换的日期 + 方向
```

## 核心概念

### scale 档位语义

| scale | 命中触发数 | 含义 |
|---:|---:|---|
| 1.0 | 0 | 满仓权益 |
| 0.6 | 1 | 60% 权益 + 40% 国债 |
| 0.4 | 2 | 40% 权益 + 60% 国债 |
| 0.3 | 3 | 30% 权益 + 70% 国债（最大防御） |

**加权平均权益暴露** ≈ `sum(scale_i) / N`，是策略整个回测期实际承担权益风险的最直接度量。

### 防御日 = scale < 1.0

`scale=1.0` 才是满仓，scale 落到 0.6/0.4/0.3 都算"防御日"。
统计里 62.5% 防御占比说明策略在多数时间里没有满仓运行——这点对业绩归因至关重要。

### "连续区间"是策略最敏感的诊断信号

- **最长连续 max-defense** 反映策略在极端市况下能不能坚持最低档（应有而不应过度）
- **最长连续满仓** 反映策略在趋势行情里能不能识别出牛市初期并加仓进攻（应长而不应过短）
- 两次档位跳变之间如果相隔很短（如 < 5 个交易日）说明触发器太敏感，反复打脸

---

## 4. AI 复盘操作手册

> **本节是给 AI 看的指令**：下次拿到一个新的 `.cache/backtest/bear_scale.csv`，
> 按下列步骤产出解读。

### 步骤 1：跑脚本拿量化数据

```bash
~/.pyenv/versions/qlib/bin/python analyze_bear_scale.py \
  --csv .cache/backtest/bear_scale.csv \
  --plot \
  --output-dir .cache/backtest/analysis/
```

输出落在 `.cache/backtest/analysis/`，**先打开 `bear_scale_report.txt` 读数字**，
再看 `bear_scale_yearly.png` 和 `bear_scale_timeline.png` 找视觉信号。

### 步骤 2：交叉对照其他回测产物

`bear_scale` 不是孤立指标，要和以下文件交叉读：

| 文件 | 用途 |
|---|---|
| `.cache/backtest/metrics.txt` | 看年化、夏普、最大回撤；判断策略整体业绩 |
| `.cache/backtest/weights.csv` | 看调仓日的实际权重，配合 bear_scale 看"防御档到底有多空仓" |
| `.cache/backtest/regime_breakdown.csv` | 看牛/震/熊分组的年化收益，配合 bear_scale 看防御档是否真的有用 |
| `.cache/backtest/nav.png` | 净值曲线上的灰底就是 bear_scale < 1.0 区间，能直观看到"防御期 vs 进攻期" |

### 步骤 3：用以下清单驱动解读

**必查的 6 个问题**：

1. **整体防御占比** 是多少？`防御天数 / 总交易日`
   - 占比 > 70% → 策略过度防御，业绩大概率被压制
   - 占比 < 30% → 防御几乎没起作用，可能熊市没识别出来
   - 50% 上下是健康区间

2. **加权平均权益暴露** 是多少？`scale.mean()`
   - < 0.5 → 策略长期处于防御状态，进攻端被压制
   - 0.7-0.8 → 健康
   - > 0.9 → 几乎不防御，熊市端回撤会大

3. **最长连续 max-defense 区间** 是哪段？持续多少天？
   - 关键诊断：策略能不能识别极端熊市并坚持最低档
   - 异常信号：max-defense 区间 < 5 天 → 触发器没真正捕获到极端情形

4. **最长连续满仓区间** 是哪段？持续多少天？
   - 关键诊断：策略能不能识别牛市并保持进攻
   - 异常信号：最长满仓 < 30 天 → 趋势行情里频繁被假信号洗出去

5. **跳变事件次数** 是多少？最近一年平均月跳变次数？
   - 健康区间：每月 1-2 次档位切换
   - > 4 次/月 → 触发器过敏感，每次小幅波动都重新调仓
   - 跳变方向分布：`defense` 多于 `attack` 说明策略更倾向于防御退出

6. **年度走势分水岭**：哪一年起防御天数显著下降？
   - 通常是熊转牛的第一年（2024→2025 或 2025→2026）
   - 如果整段回测防御占比都没下去过 → 策略可能系统性误判

### 步骤 4：写解读文档

把发现写到 `.cache/backtest/bear_scale_analysis_<start>-<end>.md`，
参考已有的 `docs/backtest-runs/bear_scale_analysis_2021-2026.md` 的章节结构：

1. 字段语义（scale 含义）
2. 整体分布（占比 + 加权暴露）
3. 按年画像（贴合 A 股节奏讲故事）
4. 关键时间窗口（max-defense / 最长满仓 / 跳变密集区）
5. 业绩归因（结合 metrics.txt）
6. 改进方向（v2 候选）

### 步骤 5：自检

- [ ] 数字与 bear_scale_report.txt 完全一致（不要重新口算）
- [ ] 中文 PNG 没有 □□ 缺字（运行脚本已自动设字体回退）
- [ ] 没有照抄上一份解读的日期（如 2022-03-30）—— 日期要随新数据更新
- [ ] 加权平均权益暴露 + 整体防御占比 是给读者最重要的两个数字

---

## 常见问题

### Q：为什么 2025-Q3 满仓天数是 66 而按年统计 2025 年满仓天数 < 65？
A：按"年"统计是日历年（YYYY），按"季度"是 (YYYY, Q)。看尺度时要注意口径。

### Q：scale=0.3 的天数为什么这么少？
A：3 触发同时命中需要：池内 ETF 站上 MA60 的 ≤ 2（极度弱势）+ 60 日波动 > 25%（高波动）+ 收盘 < MA120（熊市确认）。
A 股历史上同时满足的窗口很短，2021-2026 这 5 年里只出现过 2022-03 ~ 2022-06 这一段。

### Q：跳变事件 130+ 次会不会太多？
A：取决于回测区间长度。1301 天里 133 次跳变 = 平均每 9.8 天一次，已经偏频繁。
如果是新一轮回测拿到了更多跳变数，要重点检查：T2（60 日波动率 > 25%）是不是被 2024-2025 反复触发了。

### Q：能不能加一个"防御区间 vs 进攻区间"分组业绩对比？
A：超出本脚本范围。需要把 scale 和 nav.csv 做时序 join，再按 scale< 1.0 / scale=1.0 分组算收益。
可以做，但建议放在 v2 脚本里，不要往本脚本塞——保持职责单一。

---

## 文件清单

| 文件 | 行数估计 | 说明 |
|---|---|---|
| `analyze_bear_scale.py` | ~350 | 主脚本 |
| `docs/analyze-bear-scale.md` | 本文件 | 使用文档 |
| `docs/backtest-runs/bear_scale_analysis_2021-2026.md` | ~150 | 2026-08-13 那次复盘的样本解读 |