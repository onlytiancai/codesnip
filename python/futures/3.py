import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime

# 策略
class SmaCross(bt.Strategy):
    def __init__(self):
        sma1 = bt.ind.SMA(period=10)
        sma2 = bt.ind.SMA(period=30)
        self.crossover = bt.ind.CrossOver(sma1, sma2)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.sell()

1
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
cerebro.addstrategy(SmaCross)
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
