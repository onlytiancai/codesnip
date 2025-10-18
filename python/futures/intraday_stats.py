import pandas as pd
import numpy as np

# === 1. 数据加载 ===
file_path = "data/C9999.XDCE.10m.20251018.csv"  # 替换为你的文件路径
df = pd.read_csv(file_path)
df['timestamps'] = pd.to_datetime(df['timestamps'])
df = df.sort_values('timestamps').reset_index(drop=True)

# === 2. 基础字段 ===
df['date'] = df['timestamps'].dt.date
df['hour'] = df['timestamps'].dt.hour
df['minute'] = df['timestamps'].dt.minute
df['return'] = df['close'].pct_change()
df['range'] = (df['high'] - df['low']) / df['open']
df['up'] = (df['close'] > df['open']).astype(int)
df['mid_price'] = (df['high'] + df['low']) / 2

# === 3. 波动与ATR ===
df['hl'] = df['high'] - df['low']
df['hc'] = abs(df['high'] - df['close'].shift(1))
df['lc'] = abs(df['low'] - df['close'].shift(1))
df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
df['atr'] = df['tr'].rolling(14, min_periods=1).mean()

# === 4. VWAP 与偏离 ===
df['cum_amount'] = df['amount'].cumsum()
df['cum_volume'] = df['volume'].cumsum()
df['vwap'] = df['cum_amount'] / df['cum_volume']
df['vwap_diff'] = (df['close'] - df['vwap']) / df['vwap']
df['vwap_slope'] = df['vwap'].diff()

# === 5. 全局统计 ===
stats = {
    "样本区间": f"{df['date'].min()} 至 {df['date'].max()}",
    "总记录数": len(df),
    "阳线比例(%)": df['up'].mean() * 100,
    "平均涨跌幅(%)": df['return'].mean() * 100,
    "波动率(%)": df['return'].std() * 100,
    "平均波动幅度(%)": df['range'].mean() * 100,
    "平均ATR": df['atr'].mean(),
    "VWAP偏离均值(%)": df['vwap_diff'].mean() * 100,
    "VWAP斜率均值": df['vwap_slope'].mean(),
    "平均成交量": df['volume'].mean(),
}

# === 6. 连续上涨/下跌规律 ===
df['direction'] = np.sign(df['return'].fillna(0))
streaks = []
current_streak, current_dir = 0, 0
for d in df['direction']:
    if d == current_dir:
        current_streak += 1
    else:
        if current_dir != 0:
            streaks.append((current_dir, current_streak))
        current_dir, current_streak = d, 1
if current_streak > 0:
    streaks.append((current_dir, current_streak))
up_streaks = [s for d, s in streaks if d > 0]
down_streaks = [s for d, s in streaks if d < 0]

# === 7. 时间规律（小时级） ===
hourly = df.groupby('hour').agg(
    平均涨跌幅=('return', 'mean'),
    波动率=('return', 'std'),
    平均ATR=('atr', 'mean'),
    平均成交量=('volume', 'mean'),
    阳线比例=('up', 'mean'),
    VWAP偏离=('vwap_diff', 'mean'),
    VWAP斜率=('vwap_slope', 'mean')
)

# === 8. 量价相关 ===
corrs = {
    "成交量 vs 涨跌幅": df['volume'].corr(df['return']),
    "成交量 vs 波动幅度": df['volume'].corr(df['range']),
    "成交量 vs ATR": df['volume'].corr(df['atr']),
    "成交量 vs 收盘价": df['volume'].corr(df['close']),
    "VWAP斜率 vs 涨跌幅": df['vwap_slope'].corr(df['return']),
}

# === 9. 每日极值与收盘位置 ===
day_groups = df.groupby('date')
df['day_high'] = day_groups['high'].transform('max')
df['day_low'] = day_groups['low'].transform('min')
df['close_pos'] = (df['close'] - df['day_low']) / (df['day_high'] - df['day_low'])
df['is_day_high'] = (df['high'] == df['day_high'])
df['is_day_low'] = (df['low'] == df['day_low'])

# 高低点出现时间
high_time = df[df['is_day_high']].groupby('hour').size()
low_time = df[df['is_day_low']].groupby('hour').size()
high_low_dist = pd.DataFrame({'高点次数': high_time, '低点次数': low_time}).fillna(0)

# === 10. 每日特征波动 ===
daily = day_groups.agg(
    日波动率=('return', 'std'),
    平均ATR=('atr', 'mean'),
    平均成交量=('volume', 'mean'),
    阳线比例=('up', 'mean'),
    收盘强度均值=('close_pos', 'mean')
)
stability = {
    "日波动率标准差": daily['日波动率'].std(),
    "成交量标准差": daily['平均成交量'].std(),
    "收盘强度标准差": daily['收盘强度均值'].std(),
}

# === 11. 波动率分位区间 ===
vol_bins = pd.qcut(df['return'].abs(), q=4, labels=["低波动", "中低", "中高", "高波动"])
vol_stats = df.groupby(vol_bins).agg(
    平均成交量=('volume', 'mean'),
    阳线比例=('up', 'mean'),
    VWAP偏离均值=('vwap_diff', 'mean')
)

# === 12. 输出报告 ===
print("="*110)
print("【日内短线策略设计数据依据报告】")
print("="*110)
for k, v in stats.items():
    print(f"{k:<20}: {v:.6f}" if isinstance(v, (int, float)) else f"{k:<20}: {v}")

print("\n【连续性统计】")
print(f"连续上涨次数: {len(up_streaks)}, 平均长度: {np.mean(up_streaks) if up_streaks else 0:.2f}")
print(f"连续下跌次数: {len(down_streaks)}, 平均长度: {np.mean(down_streaks) if down_streaks else 0:.2f}")

print("\n【时间规律 - 按小时】")
for hour, row in hourly.iterrows():
    print(f"{hour:02d}:00 | 涨跌 {row['平均涨跌幅']*100:.3f}% | 波动率 {row['波动率']*100:.3f}% | "
          f"ATR {row['平均ATR']:.4f} | 成交量 {row['平均成交量']:.0f} | "
          f"阳线 {row['阳线比例']*100:.2f}% | VWAP偏离 {row['VWAP偏离']*100:.3f}% | VWAP斜率 {row['VWAP斜率']:.6f}")

print("\n【量价相关系数】")
for k, v in corrs.items():
    print(f"{k:<25}: {v:.4f}")

print("\n【极值出现时段分布】（用于判断高低点时间密度）")
for hour in range(24):
    high_cnt = high_low_dist.loc[hour, '高点次数'] if hour in high_low_dist.index else 0
    low_cnt = high_low_dist.loc[hour, '低点次数'] if hour in high_low_dist.index else 0
    print(f"{hour:02d}:00 | 高点次数 {int(high_cnt):3d} | 低点次数 {int(low_cnt):3d}")

print("\n【波动率分位特征】")
for idx, row in vol_stats.iterrows():
    print(f"{idx}: 成交量 {row['平均成交量']:.0f}, 阳线 {row['阳线比例']*100:.2f}%, VWAP偏离 {row['VWAP偏离均值']*100:.3f}%")

print("\n【稳定性指标】（衡量特征是否稳健）")
for k, v in stability.items():
    print(f"{k:<20}: {v:.6f}")

print("="*110)
