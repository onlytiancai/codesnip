import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime

# ------------------------
# 自定义 VWAP 指标（滑窗 VWAP）
# ------------------------
class VWAP(bt.Indicator):
    """
    Rolling VWAP over a window of `period` bars:
      VWAP = sum(typical_price * volume) / sum(volume)
    typical_price = (high + low + close) / 3
    """
    lines = ('vwap',)
    params = dict(period=20)

    def __init__(self):
        p = self.p.period
        # 需要至少 period 根 bar 才能输出
        self.addminperiod(p)

    def next(self):
        p = self.p.period
        # 计算过去 p 根 K 线的 numerator 和 denominator
        # 注意：range(p) 的 -i 中 -0 等于 0（当前bar），-1为上一个bar，以此类推
        tp_vol_pairs = []
        for i in range(p):
            # index = -i  -> 0, -1, -2, ...
            idx = -i
            typical = (self.data.high[idx] + self.data.low[idx] + self.data.close[idx]) / 3.0
            vol = self.data.volume[idx]
            tp_vol_pairs.append((typical, vol))

        numerator = sum(tp * vol for tp, vol in tp_vol_pairs)
        denominator = sum(vol for _, vol in tp_vol_pairs)
        if denominator != 0:
            self.lines.vwap[0] = numerator / denominator
        else:
            # 若 denominator 为 0，保持上一个值（若没有则设为 current close）
            try:
                self.lines.vwap[0] = self.lines.vwap[-1]
            except Exception:
                self.lines.vwap[0] = self.data.close[0]

# ------------------------
# 策略：VWAP + ATR 反转（并包含缩量/波动条件）
# ------------------------
class VWAPATRStrategy(bt.Strategy):
    params = dict(
        atr_period=14,
        vol_period=20,
        vwap_period=20,
        k=0.7,
        reversion_threshold=0.2,
        atr_vol_sma_period=20,  # 用来判定低/高波动的 ATR 均线周期
    )

    def __init__(self):
        # 指标
        self.atr = bt.indicators.ATR(period=self.p.atr_period)
        self.vwap = VWAP(self.data, period=self.p.vwap_period)   # 使用自定义 VWAP
        self.avg_vol = bt.indicators.SMA(self.data.volume, period=self.p.vol_period)
        self.atr_sma = bt.indicators.SMA(self.atr, period=self.p.atr_vol_sma_period)

    def next(self):
        close = float(self.data.close[0])
        vwap = float(self.vwap[0])
        atr = float(self.atr[0])
        volume = float(self.data.volume[0])
        avg_vol = float(self.avg_vol[0])
        atr_mean = float(self.atr_sma[0])

        # 检查指标是否就绪（避免 NaN）
        if any(x is None or (isinstance(x, float) and pd.isna(x)) for x in [vwap, atr, avg_vol, atr_mean]):
            return

        # 市场状态判定
        is_low_volume = volume < avg_vol
        is_low_volatility = atr < atr_mean
        is_high_volatility = atr > atr_mean

        shrink_low_vol = is_low_volume and is_low_volatility
        shrink_high_vol = is_low_volume and is_high_volatility

        # 触发条件：缩量 + (低波动或高波动)
        if not (shrink_low_vol or shrink_high_vol):
            return

        # 入场
        if not self.position:
            if close > vwap + self.p.k * atr:
                # 做空
                self.sell()
                self.log(f"SHORT ENTRY  close={close:.2f} vwap={vwap:.2f} atr={atr:.4f}")
            elif close < vwap - self.p.k * atr:
                # 做多
                self.buy()
                self.log(f"LONG ENTRY   close={close:.2f} vwap={vwap:.2f} atr={atr:.4f}")

        # 出场：多单
        elif self.position.size > 0:
            # 回归至 VWAP 附近
            if close >= vwap - self.p.reversion_threshold * atr:
                self.close()
                self.log(f"LONG EXIT (revert) close={close:.2f} vwap={vwap:.2f}")
            # 或出现反向信号（价格位于上方做空区间）
            elif close > vwap + self.p.k * atr:
                self.close()
                self.log(f"LONG EXIT (reverse) close={close:.2f}")

        # 出场：空单
        elif self.position.size < 0:
            if close <= vwap + self.p.reversion_threshold * atr:
                self.close()
                self.log(f"SHORT EXIT (revert) close={close:.2f} vwap={vwap:.2f}")
            elif close < vwap - self.p.k * atr:
                self.close()
                self.log(f"SHORT EXIT (reverse) close={close:.2f}")

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')

# ================== 主程序：载入数据、运行回测、导出 ==================
def prepare_bt_csv(src_csv, tmp_csv):
    # 读取原CSV并将 'timestamps,open,high,low,close,volume,amount' 转为 backtrader-friendly CSV
    # backtrader GenericCSVData: datetime,open,high,low,close,volume,openinterest
    df = pd.read_csv(src_csv, parse_dates=['timestamps'])
    df_bt = pd.DataFrame()
    df_bt['datetime'] = df['timestamps'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_bt['open'] = df['open']
    df_bt['high'] = df['high']
    df_bt['low'] = df['low']
    df_bt['close'] = df['close']
    df_bt['volume'] = df['volume']
    # 将 amount 放到 openinterest 列以便在策略中访问（简便方法）
    df_bt['openinterest'] = df['amount']
    df_bt.to_csv(tmp_csv, index=False)
CSV_PATH = "data/C9999.XDCE.10m.20251018.csv"  # 你的 CSV 路径
TMP_BT_CSV = CSV_PATH + ".bt.csv"
prepare_bt_csv(CSV_PATH, TMP_BT_CSV)
data_bt = bt.feeds.GenericCSVData(
    dataname=TMP_BT_CSV,
    dtformat='%Y-%m-%d %H:%M:%S',
    datetime=0, open=1, high=2, low=3, close=4, volume=5, openinterest=6,
    timeframe=bt.TimeFrame.Minutes, compression=10  # 10-minute bars
)



# 创建引擎
cerebro = bt.Cerebro()
cerebro.adddata(data_bt)
cerebro.addstrategy(VWAPATRStrategy)
cerebro.broker.set_cash(10000)
cerebro.broker.setcommission(commission=0.001)

# 🔍 分析器（针对10分钟数据）
# 默认 SharpeRatio 假设每日数据，因此要自定义 annualize 倍数
# 如果你认为每天有6.5小时（39根10分钟K线），一年252个交易日：
ANNUAL_FACTOR = 39 * 252  # ≈ 9828 根K线/年

cerebro.addanalyzer(
    bt.analyzers.SharpeRatio,
    _name='sharpe',
    timeframe=bt.TimeFrame.Minutes,
    annualize=True,
    factor=ANNUAL_FACTOR
)
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

# 运行回测
print("开始回测...")
results = cerebro.run()
print("回测完成。")

# 结果分析（安全处理）
strat = results[0]
analyzers = strat.analyzers

sharpe = analyzers.sharpe.get_analysis().get('sharperatio', None)
drawdown = analyzers.drawdown.get_analysis()
trade = analyzers.trade.get_analysis()
returns = analyzers.returns.get_analysis()

def safe_get(d, *keys, default=0):
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            cur = getattr(cur, k, default)
    return cur

total_trades = safe_get(trade, 'total', 'total', default=0)
won_trades = safe_get(trade, 'won', 'total', default=0)
lost_trades = safe_get(trade, 'lost', 'total', default=0)
win_rate = (won_trades / total_trades * 100) if total_trades else 0

print("\n📊 回测结果指标汇总")
print("=" * 40)
print(f"最终资金:       {cerebro.broker.getvalue():.2f} USD")
print(f"总收益率:       {returns.get('rtot', 0)*100:.2f}%")
print(f"年化收益率:     {returns.get('rnorm100', 0):.2f}%")
print(f"夏普比率:       {sharpe if sharpe else 'N/A'}")
print(f"最大回撤:       {drawdown['max']['drawdown']:.2f}%")
print(f"交易次数:       {total_trades}")
print(f"胜率:           {win_rate:.2f}%")
print("=" * 40)

cerebro.plot()
