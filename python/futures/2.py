# intraday_bt_full.py
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import time
import os

plt.rcParams['font.sans-serif'] = ['Heiti TC']   # macOS 默认中文系统字体
plt.rcParams['axes.unicode_minus'] = False           # 正常显示负号

# ================== 配置参数 ==================
CSV_PATH = "data/C9999.XDCE.10m.20251018.csv"  # 你的 CSV 路径
TMP_BT_CSV = CSV_PATH + ".bt.csv"

VOL_CHG_THRESHOLD = 0.5      # 成交量放大阈值
ATR_PERIOD = 10
ATR_MULT_HIGH = 1.2
ATR_MULT_LOW = 0.8
MOM_PERIOD = 3
VWAP_K = 0.7
STOPLOSS_ATR = 2.0
TAKEPROFIT_ATR = 3.0
MAX_POS_RISK = 0.005        # 单笔风险占权益比例
MAX_OPEN_POS = 2

TRADE_SESSIONS = [
    (time(21,0), time(23,0)),
    (time(9,0), time(10,15)),
    (time(10,30), time(11,30)),
    (time(13,30), time(15,0))
]

def in_trade_session(t: time) -> bool:
    for s,e in TRADE_SESSIONS:
        if s <= t < e:
            return True
    return False

# ================== 策略实现 ==================
class VolumeATRStrategy(bt.Strategy):
    params = dict(
        vol_chg_thresh=VOL_CHG_THRESHOLD,
        atr_period=ATR_PERIOD,
        atr_high_mul=ATR_MULT_HIGH,
        atr_low_mul=ATR_MULT_LOW,
        mom_period=MOM_PERIOD,
        vwap_k=VWAP_K,
        stop_atr=STOPLOSS_ATR,
        take_atr=TAKEPROFIT_ATR,
        max_pos_risk=MAX_POS_RISK,
        max_open_pos=MAX_OPEN_POS
    )

    def __init__(self):
        # 指标
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        # 存储上一bar成交量，用于计算 vol_chg
        self.last_volume = None
        # 用 EWMA 跟踪 ATR 的均值（用于阈值判断）
        self.atr_mean = None

        # 用于记录交易信息（逐笔）
        self.open_execs = []   # list of dicts for currently opened executions (waiting to be closed)
        self.trades = []       # finished trade records
        self._trade_id = 0

        # 存储上次计算的 vol_chg 与 market state，方便在 notify_order 时记录
        self.last_vol_chg = 0.0
        self.last_state = 'N/A'

    def log(self, txt):
        dt = self.data.datetime.datetime(0)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}  {txt}")

    def next(self):
        dt = self.data.datetime.datetime(0)
        t = dt.time()

        # 读取 amount（成交额）和 volume，计算本bar的 VWAP（若 amount 可用）
        # 注意：我们在准备数据时把 amount 写到了 openinterest 列，读取时为 self.data.openinterest
        try:
            amount = float(self.data.openinterest[0])
            vol = float(self.data.volume[0])
            if vol > 0:
                cur_vwap = amount / vol
            else:
                cur_vwap = float(self.data.close[0])
        except Exception:
            cur_vwap = float(self.data.close[0])
            vol = float(self.data.volume[0])

        # 更新 ATR 均值 (EWMA)
        if self.atr[0] is not None:
            if self.atr_mean is None:
                self.atr_mean = float(self.atr[0])
            else:
                self.atr_mean = 0.05 * float(self.atr[0]) + 0.95 * self.atr_mean

        # 非交易时段：平掉持仓并跳过信号
        if not in_trade_session(t):
            if self.position:
                self.close()
                self.log(f"Close (session end) size={self.position.size} price={self.data.close[0]:.2f}")
            # 更新 last_volume 并返回
            self.last_volume = vol
            return

        # 计算成交量变化率
        vol_chg = 0.0
        if self.last_volume is not None and self.last_volume > 0:
            vol_chg = vol / self.last_volume - 1.0
        self.last_volume = vol
        self.last_vol_chg = vol_chg  # 保存以便记录

        # 没有足够 ATR 数据时跳过
        if self.atr_mean is None or self.atr[0] is None:
            return

        # 判断市场状态
        is_vol_up = vol_chg > self.p.vol_chg_thresh
        is_atr_high = self.atr[0] > (self.atr_mean * self.p.atr_high_mul)
        is_atr_low = self.atr[0] < (self.atr_mean * self.p.atr_low_mul)

        if is_vol_up and is_atr_high:
            state = '放量高波动'
        elif is_vol_up and not is_atr_high:
            state = '放量低波动'
        elif not is_vol_up and is_atr_high:
            state = '缩量高波动'
        else:
            state = '缩量低波动'
        self.last_state = state

        # 检查并执行平仓（TP/SL）
        if self.position:
            entry_price = self.position.price if hasattr(self.position, 'price') else None
            # 使用 position.price 可能需要根据具体sizer/backtrader行为调整
            if entry_price is None:
                entry_price = self.position.price
            # 多头
            if self.position.size > 0:
                stop = entry_price - self.p.stop_atr * self.atr[0]
                take = entry_price + self.p.take_atr * self.atr[0]
                if self.data.close[0] <= stop or self.data.close[0] >= take:
                    self.log(f"Close (TP/SL) size={self.position.size} price={self.data.close[0]:.2f} state={state}")
                    self.close()
                    return
            else:
                # 空头
                stop = entry_price + self.p.stop_atr * self.atr[0]
                take = entry_price - self.p.take_atr * self.atr[0]
                if self.data.close[0] >= stop or self.data.close[0] <= take:
                    self.log(f"Close (TP/SL short) size={self.position.size} price={self.data.close[0]:.2f} state={state}")
                    self.close()
                    return

        # 限制并行仓位数量
        # use len(self.broker.positions) not always ideal; instead count active trades recorded
        active_positions = 1 if self.position else 0
        if active_positions >= self.p.max_open_pos:
            return

        # 生成信号
        signal = None
        # 动量模式
        if state == '放量高波动':
            if len(self.data.close) > self.p.mom_period:
                mom = self.data.close[0] - self.data.close[-self.p.mom_period]
                if mom > 0 and vol_chg > self.p.vol_chg_thresh:
                    signal = 'LONG'
                elif mom < 0 and vol_chg > self.p.vol_chg_thresh:
                    signal = 'SHORT'
        # 均值回归模式
        elif state in ['缩量低波动', '缩量高波动']:
            diff = float(self.data.close[0]) - cur_vwap
            if diff > self.p.vwap_k * self.atr[0]:
                signal = 'SHORT'
            elif diff < -self.p.vwap_k * self.atr[0]:
                signal = 'LONG'
        # 放量低波动：保守或可定义更严格动量条件（此处跳过）

        # 下单（按照风险计算合约数量）
        if signal is not None:
            price = float(self.data.close[0])
            equity = float(self.broker.getvalue())

            if signal == 'LONG':
                stop_price = price - self.p.stop_atr * self.atr[0]
                risk_per_unit = price - stop_price
            else:
                stop_price = price + self.p.stop_atr * self.atr[0]
                risk_per_unit = stop_price - price

            if risk_per_unit <= 0:
                return

            size = int((equity * self.p.max_pos_risk) / risk_per_unit)
            if size <= 0:
                return

            # 下单并记录执行上下文（在 notify_order 完成时保存）
            if signal == 'LONG':
                self.buy(size=size)
                self.log(f"BUY signal size={size} price={price:.2f} vol_chg={vol_chg:.3f} ATR={self.atr[0]:.3f} state={state}")
            else:
                self.sell(size=size)
                self.log(f"SELL signal size={size} price={price:.2f} vol_chg={vol_chg:.3f} ATR={self.atr[0]:.3f} state={state}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            # 记录执行（作为 open_execs 的一项）
            executed_dt = self.data.datetime.datetime(0)
            o = {
                'open_time': executed_dt,
                'entry_price': order.executed.price,
                'size': order.executed.size,
                'side': 'LONG' if order.isbuy() else 'SHORT',
                'vol_chg': getattr(self, 'last_vol_chg', 0.0),
                'state': getattr(self, 'last_state', 'N/A'),
            }
            self.open_execs.append(o)
            self.log(f"Order Executed {'BUY' if order.isbuy() else 'SELL'} size={o['size']} price={o['entry_price']:.2f}")

        elif order.status in [order.Canceled, order.Rejected]:
            self.log("Order Canceled/Rejected")

    def notify_trade(self, trade):
        # 当一个trade 完全平仓（isclosed）时记录逐笔信息
        if trade.isclosed:
            # pop earliest open_exec (FIFO) - reasonable for simple single-instrument sequential trades
            if len(self.open_execs) == 0:
                # 没有对应记录则用当前数据填充
                open_meta = {
                    'open_time': None, 'entry_price': None, 'size': trade.size,
                    'side': 'LONG' if trade.size > 0 else 'SHORT', 'vol_chg': None, 'state': None
                }
            else:
                open_meta = self.open_execs.pop(0)

            close_time = self.data.datetime.datetime(0)
            entry_price = open_meta.get('entry_price', None) if open_meta else None
            exit_price = trade.price if trade.price is not None else float(self.data.close[0])
            pnl = trade.pnl  # 未扣手续费的净盈亏
            pnl_comm = trade.pnlcomm  # 扣手续费后的
            record = {
                'trade_id': self._trade_id,
                'open_time': open_meta.get('open_time', None),
                'close_time': close_time,
                'side': open_meta.get('side', 'N/A'),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': trade.size,
                'pnl': pnl,
                'pnl_comm': pnl_comm,
                'vol_chg': open_meta.get('vol_chg', None),
                'state': open_meta.get('state', None),
                'atr_at_close': float(self.atr[0]) if self.atr[0] is not None else None
            }
            self.trades.append(record)
            self._trade_id += 1
            self.log(f"Trade closed id={self._trade_id-1} pnl={pnl:.2f}")

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

def run_backtest():
    # Prepare temp CSV for backtrader
    prepare_bt_csv(CSV_PATH, TMP_BT_CSV)

    cerebro = bt.Cerebro(stdstats=True)
    cerebro.addstrategy(VolumeATRStrategy)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')

    data = bt.feeds.GenericCSVData(
        dataname=TMP_BT_CSV,
        dtformat='%Y-%m-%d %H:%M:%S',
        datetime=0, open=1, high=2, low=3, close=4, volume=5, openinterest=6,
        timeframe=bt.TimeFrame.Minutes, compression=10  # 10-minute bars
    )
    cerebro.adddata(data)
    cerebro.broker.setcash(1_000_000.0)
    # 真实回测应设置手续费、滑点、合约乘数等
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)

    print("Start Value:", cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]
    print("End Value:", cerebro.broker.getvalue())

    # 导出逐笔交易明细
    trades_df = pd.DataFrame(strat.trades)
    if not trades_df.empty:
        # 转换时间列为字符串
        trades_df['open_time'] = trades_df['open_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x is not None else None)
        trades_df['close_time'] = trades_df['close_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x is not None else None)
    trades_df.to_csv('backtest_trades.csv', index=False)
    print(f"已导出逐笔交易 -> backtest_trades.csv (共 {len(trades_df)} 条)")

    # 分象限统计（基于记录中的 vol_chg 与 atr_at_close）
    if not trades_df.empty:
        def classify_row(r):
            vol_chg = r['vol_chg'] if pd.notnull(r['vol_chg']) else 0.0
            atr_val = r['atr_at_close'] if pd.notnull(r['atr_at_close']) else strat.atr_mean if strat.atr_mean is not None else 0.0
            is_vol_up = vol_chg > VOL_CHG_THRESHOLD
            is_atr_high = atr_val > strat.atr_mean * ATR_MULT_HIGH if strat.atr_mean is not None else False
            if is_vol_up and is_atr_high:
                return '放量高波动'
            elif is_vol_up and not is_atr_high:
                return '放量低波动'
            elif not is_vol_up and is_atr_high:
                return '缩量高波动'
            else:
                return '缩量低波动'

        trades_df['quadrant'] = trades_df.apply(classify_row, axis=1)

        quad_stats = trades_df.groupby('quadrant')['pnl'].agg(['count', 'mean', 'sum']).reindex(
            ['放量高波动','放量低波动','缩量高波动','缩量低波动']
        ).fillna(0)
        quad_stats.to_csv('quadrant_stats.csv')
        print("已导出分象限统计 -> quadrant_stats.csv")
        print(quad_stats)

        # 画象限平均单笔收益条形图
        plt.figure(figsize=(8,4))
        quad_stats['mean'].plot(kind='bar')
        plt.title('不同量价象限的平均单笔收益')
        plt.ylabel('平均PNL')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('quadrant_pnl_bar.png', dpi=150)
        print("象限收益图 -> quadrant_pnl_bar.png")

        # 累计收益曲线（按平仓时间顺序）
        trades_df = trades_df.sort_values('close_time')
        trades_df['cum_pnl'] = trades_df['pnl'].cumsum()
        plt.figure(figsize=(10,4))
        plt.plot(pd.to_datetime(trades_df['close_time']), trades_df['cum_pnl'], marker='o', markersize=3)
        plt.title('逐笔累计收益曲线')
        plt.xlabel('时间')
        plt.ylabel('累计PNL')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('equity_curve.png', dpi=150)
        print("收益曲线图 -> equity_curve.png")
    else:
        print("无交易记录，未生成象限统计与图表。")

    # 清理临时文件
    try:
        os.remove(TMP_BT_CSV)
    except Exception:
        pass

   
    analyzers = strat.analyzers
    print('Sharpe Ratio:', analyzers.sharpe.get_analysis())
    print('Max Drawdown:', analyzers.drawdown.get_analysis())
    print('Trade Stats:', analyzers.trade.get_analysis())
    
    sharpe = analyzers.sharpe.get_analysis()
    drawdown = analyzers.drawdown.get_analysis()
    trade_stats = analyzers.trade.get_analysis()

    total_return = (cerebro.broker.getvalue() / 1_000_000) - 1
    max_drawdown = drawdown.max.drawdown / 100 if hasattr(drawdown, 'max') else 0
    sharpe_ratio = sharpe.get('sharperatio', 0) or 0
    win_rate = trade_stats.won.total / trade_stats.total.closed if trade_stats.total.closed else 0

    print("\n=== 策略绩效指标 ===")
    print(f"总收益率: {total_return*100:.2f}%")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print(f"胜率: {win_rate*100:.2f}%")
    print(f"夏普率: {sharpe_ratio:.2f}")

    perf_summary = pd.DataFrame([{
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': len(trades_df)
    }])
    perf_summary.to_csv('performance_summary.csv', index=False)
    print("已导出策略绩效指标 -> performance_summary.csv")
    
if __name__ == '__main__':
    run_backtest()
