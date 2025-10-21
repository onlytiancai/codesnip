import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 设置中文字体（适用于 macOS） =====
plt.rcParams['font.sans-serif'] = ['Heiti TC']   # macOS 默认中文系统字体
plt.rcParams['axes.unicode_minus'] = False           # 正常显示负号

# === 1. 读取数据 ===
file_path = "data/C9999.XDCE.10m.20251018.csv"  # 修改为你的路径
df = pd.read_csv(file_path, parse_dates=['timestamps'])
df.set_index('timestamps', inplace=True)
df = df.sort_index()

# === 2. 构造衍生特征 ===
df['ret'] = df['close'].pct_change() # 收盘价收益率
df['hl_range'] = df['high'] - df['low'] # 10分钟波动幅度
df['oc_range'] = df['close'] - df['open'] # 开盘到收盘的方向与强度
df['vwap'] = df['amount'] / df['volume']# 成交均价
df['abs_ret'] = df['ret'].abs()


# 时间字段
df['hour'] = df.index.hour
df['minute'] = df.index.minute
df['time_bin'] = df['hour'] * 60 + df['minute']
df['date'] = df.index.date

# === 3. 时段收益统计 ===
# time_bin 已是分钟数，可转换为时:分格式方便阅读
time_stats = df.groupby('time_bin')['ret'].agg(['mean', 'std', 'count'])

# 将 time_bin 转成 "H:M" 格式显示
def minute_to_hm(m):
    hour = m // 60
    minute = m % 60
    return f"{int(hour):02d}:{int(minute):02d}"

time_stats.index = time_stats.index.map(minute_to_hm)

# 打印时段统计信息（前几行）
print("=== 不同时间段平均收益率统计 ===")
print(time_stats.head(20))
print("\n共计时段数：", len(time_stats))

import matplotlib.dates as mdates
from datetime import time
# 绘图
# === 3. 定义玉米期货交易时段 ===
trading_sessions = [
    (time(21, 0), time(23, 0)),    # 夜盘
    (time(9, 0), time(10, 15)),    # 日盘 1
    (time(10, 30), time(11, 30)),  # 日盘 2
    (time(13, 30), time(15, 0))    # 日盘 3
]

def is_trading_time(t):
    """判断时间是否在交易时段内"""
    for start, end in trading_sessions:
        if start <= t.time() < end:
            return True
    return False

# === 4. 筛选交易时段数据 ===
df_trading = df[df.index.map(is_trading_time)].copy()

# === 5. 按时间（不含日期）统计收益 ===
df_trading['time_str'] = df_trading.index.time.astype(str)
time_stats = df_trading.groupby('time_str')['ret'].agg(['mean', 'std', 'count'])

# 将字符串索引转换为datetime，方便排序与绘图
time_stats.index = pd.to_datetime(time_stats.index, format='%H:%M:%S')
time_stats = time_stats.sort_index()

# === 6. 绘制时段平均收益率图 ===
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(time_stats.index, time_stats['mean'] * 100, marker='o', markersize=2, label='平均收益率')

# 时间格式化为 %H:%M
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())

plt.title('玉米期货各交易时段平均收益率（%）')
plt.xlabel('时间')
plt.ylabel('平均收益率（%）')
plt.grid(True)
plt.xticks(rotation=45)

# 添加交易时段分界线
for t in ['23:00', '10:15', '11:30', '15:00']:
    ax.axvline(pd.to_datetime(t, format='%H:%M'), color='gray', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.legend()
# plt.show()

# === 4. 趋势延续性 ===
autocorr_1 = df['ret'].autocorr(lag=1)
print(f"收益率 1期自相关: {autocorr_1:.4f}")

# 连涨/跌统计
df['ret_sign'] = np.sign(df['ret'])
runs = (df['ret_sign'] != df['ret_sign'].shift()).cumsum()
run_stats = df.groupby(runs)['ret_sign'].agg(['count','first']).value_counts().unstack()
print("\n连续上涨/下跌段统计：")
print(run_stats)

# === 5. 波动与成交量关系 ===
corr_v = df[['abs_ret','volume']].corr().iloc[0,1]
print(f"\n波动与成交量相关系数: {corr_v:.3f}")

# 按成交量分组看波动率
vol_group = df.groupby(pd.qcut(df['volume'], 10), observed=True)['abs_ret'].std()
# 打印成交量分组与波动率统计
print("\n=== 成交量分位数与波动率统计 ===")
for interval, val in vol_group.items():
    left = int(interval.left)
    right = int(interval.right)
    print(f"成交量区间 {left:,} ~ {right:,} : 波动率 = {val:.6f}")


# === 6. 大涨/跌后的均值回归 ===
quant_up = df['ret'].quantile(0.99)
quant_down = df['ret'].quantile(0.01)
future_mean_up = df.loc[df['ret']>quant_up, 'ret'].shift(-1).mean()
future_mean_down = df.loc[df['ret']<quant_down, 'ret'].shift(-1).mean()

print(f"\n大涨后下一bar平均收益: {future_mean_up:.5f}")
print(f"大跌后下一bar平均收益: {future_mean_down:.5f}")

# === 7. 波动聚集性 (Volatility Clustering) ===
for lag in [1,2,3,5,10]:
    ac = df['abs_ret'].autocorr(lag)
    print(f"abs_ret 自相关 (lag={lag}): {ac:.4f}")

# === 8

# === 构造衍生指标 ===
df['vwap_diff'] = df['close'] - df['vwap']               # VWAP 偏离
df['vol_chg'] = df['volume'] / df['volume'].shift(1) - 1 # 成交量变化率
df['momentum_3'] = df['close'].diff(3)                   # 3周期动量
df['momentum_6'] = df['close'].diff(6)                   # 6周期动量
df['atr'] = df['hl_range'].rolling(10).mean()            # 10个bar的平均波动（类ATR）

# === 打印主要统计描述 ===
print("\n=== 主要量价与动能指标描述性统计 ===")
print(df[['vwap_diff', 'vol_chg', 'momentum_3', 'momentum_6', 'atr']].describe().T.round(6))

# === 1️⃣ VWAP偏离与未来收益关系 ===
df['future_ret'] = df['close'].shift(-1) / df['close'] - 1
corr_vwap = df['vwap_diff'].corr(df['future_ret'])
print(f"\nVWAP偏离与下一bar收益相关系数: {corr_vwap:.4f}")

# VWAP分组分析（看均值回归倾向）
vwap_group = df.groupby(pd.qcut(df['vwap_diff'], 10), observed=True)['future_ret'].mean()
print("\nVWAP偏离分位数与下一bar平均收益：")
for interval, val in vwap_group.items():
    left = interval.left
    right = interval.right
    print(f"VWAP偏离区间 [{left:.4f}, {right:.4f}] → 平均未来收益: {val:.6f}")

# === 2️⃣ 成交量变化率与波动的关系 ===
corr_vol_vol = df['vol_chg'].corr(df['abs_ret'])
print(f"\n成交量变化率与当期波动(abs_ret)相关系数: {corr_vol_vol:.4f}")

volchg_group = df.groupby(pd.qcut(df['vol_chg'], 10), observed=True)['abs_ret'].mean()
print("\n成交量变化率分位数与平均波动率：")
for interval, val in volchg_group.items():
    print(f"变化率区间 [{interval.left:.2f}, {interval.right:.2f}] → 平均波动: {val:.6f}")

# === 3️⃣ 动量指标与未来收益关系 ===
corr_mom3 = df['momentum_3'].corr(df['future_ret'])
corr_mom6 = df['momentum_6'].corr(df['future_ret'])
print(f"\n动量指标与下一bar收益相关系数：momentum_3={corr_mom3:.4f}, momentum_6={corr_mom6:.4f}")

# 动量分组收益
mom_group = df.groupby(pd.qcut(df['momentum_3'], 10, duplicates='drop'), observed=True)['future_ret'].mean()
print("\n3周期动量分位数与未来收益：")
for interval, val in mom_group.items():
    print(f"动量区间 [{interval.left:.2f}, {interval.right:.2f}] → 平均未来收益: {val:.6f}")

# === 4️⃣ ATR与波动聚集关系 ===
corr_atr_vol = df['atr'].corr(df['abs_ret'])
print(f"\nATR与即时波动(abs_ret)相关系数: {corr_atr_vol:.4f}")

# ATR分组波动
atr_group = df.groupby(pd.qcut(df['atr'].dropna(), 10), observed=True)['abs_ret'].mean()
print("\nATR分位数与当期波动率：")
for interval, val in atr_group.items():
    print(f"ATR区间 [{interval.left:.4f}, {interval.right:.4f}] → 平均波动: {val:.6f}")