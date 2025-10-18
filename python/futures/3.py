import backtrader as bt
import yfinance as yf
import pandas as pd   # ✅ 加上这一行
from datetime import datetime

class SmaCross(bt.Strategy):
    def __init__(self):
        sma1 = bt.ind.SMA(period=10)
        sma2 = bt.ind.SMA(period=30)
        self.crossover = bt.ind.CrossOver(sma1, sma2)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.sell()

# 下载数据
data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")

# ✅ 关键：扁平化列名，避免 tuple 错误
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# 转为 backtrader 数据格式
data_bt = bt.feeds.PandasData(dataname=data)

# 运行回测
cerebro = bt.Cerebro()
cerebro.adddata(data_bt)
cerebro.addstrategy(SmaCross)
cerebro.broker.set_cash(10000)
cerebro.run()
cerebro.plot()
