import pandas as pd
import numpy as np

# === 1. 数据加载 ===
file_path = "data/C9999.XDCE.10m.20251018.csv"  # 修改为你的文件路径
df = pd.read_csv(file_path)
df['timestamps'] = pd.to_datetime(df['timestamps'])
df = df.sort_values('timestamps').reset_index(drop=True)

# === 2. 基础处理 ===
df['date'] = df['timestamps'].dt.date
df['hour'] = df['timestamps'].dt.hour
df['return'] = df['close'].pct_change().fillna(0)
df['range'] = (df['high'] - df['low']) / df['open']
df['up'] = (df['close'] > df['open']).astype(int)

# === 3. 真实波动ATR ===
df['hl'] = df['high'] - df['low']
df['hc'] = abs(df['high'] - df['close'].shift(1))
df['lc'] = abs(df['low'] - df['close'].shift(1))
df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
df['atr'] = df['tr'].rolling(14, min_periods=1).mean()

# === 4. VWAP相关 ===
df['cum_amount'] = df['amount'].cumsum()
df['cum_volume'] = df['volume'].cumsum().replace(0, np.nan)
df['vwap'] = df['cum_amount'] / df['cum_volume']
df['vwap_diff'] = (df['close'] - df['vwap']) / df['vwap']
df['vwap_slope'] = df['vwap'].diff()
df['vwap_sign'] = np.sign(df['vwap_slope'])

# === 5. 全局统计 ===
stats = {
    "样本区间": f"{df['date'].min()} 至 {df['date'].max()}",
    "记录数": len(df),
    "阳线比例(%)": df['up'].mean() * 100,
    "平均涨跌幅(%)": df['return'].mean() * 100,
    "波动率(%)": df['return'].std() * 100,
    "平均波动幅度(%)": df['range'].mean() * 100,
    "平均ATR": df['atr'].mean(),
    "VWAP偏离均值(%)": df['vwap_diff'].mean() * 100,
    "VWAP斜率均值": df['vwap_slope'].mean(),
    "平均成交量": df['volume'].mean(),
}

# === 6. 趋势结构 ===
df['direction'] = np.sign(df['return'])
# 连续段长度统计
streaks = []
cur_streak, cur_dir = 0, 0
for d in df['direction']:
    if d == cur_dir:
        cur_streak += 1
    else:
        if cur_dir != 0:
            streaks.append((cur_dir, cur_streak))
        cur_dir, cur_streak = d, 1
if cur_streak > 0:
    streaks.append((cur_dir, cur_streak))
up_streaks = [s for d, s in streaks if d > 0]
down_streaks = [s for d, s in streaks if d < 0]

# === 7. 方向转移矩阵 ===
dirs = df['direction'].shift(1)
matrix = pd.crosstab(dirs, df['direction'], normalize='index').fillna(0)
matrix.index = matrix.index.map({-1: '前一周期下跌', 0: '前一周期平盘', 1: '前一周期上涨'})
matrix.columns = ['下跌', '平盘', '上涨']

# === 8. VWAP趋势分型统计 ===
vwap_trend = df.groupby('vwap_sign').agg(
    平均涨跌幅=('return', 'mean'),
    阳线比例=('up', 'mean'),
    平均ATR=('atr', 'mean'),
    VWAP偏离均值=('vwap_diff', 'mean')
)
vwap_trend.index = vwap_trend.index.map({-1: 'VWAP下降阶段', 0: 'VWAP平缓阶段', 1: 'VWAP上升阶段'})

# === 9. 时间规律（按小时） ===
hourly = df.groupby('hour').agg(
    平均涨跌幅=('return', 'mean'),
    波动率=('return', 'std'),
    平均ATR=('atr', 'mean'),
    成交量均值=('volume', 'mean'),
    阳线比例=('up', 'mean'),
    VWAP偏离=('vwap_diff', 'mean')
)

# === 10. 量价相关性 ===
corrs = {
    "成交量 vs 涨跌幅": df['volume'].corr(df['return']),
    "成交量 vs 波动幅度": df['volume'].corr(df['range']),
    "成交量 vs ATR": df['volume'].corr(df['atr']),
    "VWAP斜率 vs 涨跌幅": df['vwap_slope'].corr(df['return']),
    "VWAP偏离 vs 涨跌幅": df['vwap_diff'].corr(df['return'])
}

# === 11. 极值分布（高低点出现时间） ===
day_groups = df.groupby('date')
df['day_high'] = day_groups['high'].transform('max')
df['day_low'] = day_groups['low'].transform('min')
df['is_high'] = (df['high'] == df['day_high'])
df['is_low'] = (df['low'] == df['day_low'])
high_time = df[df['is_high']].groupby('hour').size()
low_time = df[df['is_low']].groupby('hour').size()
high_low = pd.DataFrame({'高点次数': high_time, '低点次数': low_time}).fillna(0)

# === 12. 跨日稳定性 ===
daily = day_groups.agg(
    日波动率=('return', 'std'),
    平均ATR=('atr', 'mean'),
    平均成交量=('volume', 'mean'),
    阳线比例=('up', 'mean'),
)
stability = {
    "波动率标准差": daily['日波动率'].std(),
    "成交量标准差": daily['平均成交量'].std(),
    "阳线比例标准差": daily['阳线比例'].std(),
}

# === 13. 波动率分位区间 ===
vol_bins = pd.qcut(df['return'].abs(), q=4, labels=["低波动", "中低", "中高", "高波动"])
vol_stats = df.groupby(vol_bins).agg(
    平均成交量=('volume', 'mean'),
    阳线比例=('up', 'mean'),
    VWAP偏离均值=('vwap_diff', 'mean')
)

# === 14. 滞后相关性（Momentum延续） ===
max_lag = 5
lag_corrs = {f"Lag {i}": df['return'].autocorr(lag=i) for i in range(1, max_lag+1)}

# === 15. 打印报告 ===
print("="*120)
print("【全维度日内短线数据统计报告】")
print("="*120)
for k, v in stats.items():
    print(f"{k:<20}: {v:.6f}" if isinstance(v, (int, float)) else f"{k:<20}: {v}")

print("\n【趋势结构】")
print(f"连续上涨次数: {len(up_streaks)} | 平均长度: {np.mean(up_streaks):.2f}")
print(f"连续下跌次数: {len(down_streaks)} | 平均长度: {np.mean(down_streaks):.2f}")

print("\n【方向转移矩阵（条件概率）】")
print(matrix.to_string(float_format=lambda x: f"{x*100:6.2f}%"))

print("\n【VWAP趋势分型统计】")
for idx, row in vwap_trend.iterrows():
    print(f"{idx:<12} | 涨跌 {row['平均涨跌幅']*100:.3f}% | 阳线 {row['阳线比例']*100:.2f}% | "
          f"ATR {row['平均ATR']:.4f} | VWAP偏离 {row['VWAP偏离均值']*100:.3f}%")

print("\n【时间规律 - 按小时】")
for hour, row in hourly.iterrows():
    print(f"{hour:02d}:00 | 涨跌 {row['平均涨跌幅']*100:.3f}% | 波动率 {row['波动率']*100:.3f}% | "
          f"ATR {row['平均ATR']:.4f} | 成交量 {row['成交量均值']:.0f} | "
          f"阳线 {row['阳线比例']*100:.2f}% | VWAP偏离 {row['VWAP偏离']*100:.3f}%")

print("\n【量价相关性】")
for k, v in corrs.items():
    print(f"{k:<25}: {v:.4f}")

print("\n【极值时段分布】")
for h in range(24):
    high_cnt = high_low.loc[h, '高点次数'] if h in high_low.index else 0
    low_cnt = high_low.loc[h, '低点次数'] if h in high_low.index else 0
    print(f"{h:02d}:00 | 高点 {int(high_cnt):3d} | 低点 {int(low_cnt):3d}")

print("\n【波动率分位特征】")
for idx, row in vol_stats.iterrows():
    print(f"{idx:<4} | 成交量 {row['平均成交量']:.0f} | 阳线 {row['阳线比例']*100:.2f}% | VWAP偏离 {row['VWAP偏离均值']*100:.3f}%")

print("\n【跨日稳定性】")
for k, v in stability.items():
    print(f"{k:<20}: {v:.6f}")

print("\n【滞后相关性（涨跌延续性）】")
for k, v in lag_corrs.items():
    print(f"{k:<8}: {v:.4f}")

print("="*120)
