#!/usr/bin/env python3
"""
SPY日内量化策略回测 v4 (重构版)
从CSV文件读取数据，支持多种时间周期

用法:
    python backtest.py --data spy_hourly.csv
    python backtest.py --data spy_1min_recent.csv --strategy vwap rsi
    python backtest.py --data spy_hourly.csv --commission 0.0003
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# ============ 指标计算 ============
def compute_rsi(prices, period=5):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_vwap(df):
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical * df['Volume']).cumsum() / df['Volume'].cumsum()

def compute_ma(prices, period):
    return prices.rolling(period).mean()

# ============ 策略 ============
class VWAPMeanReversion:
    """VWAP偏离均值回归策略"""
    def __init__(self, entry_threshold=0.015):
        self.entry_threshold = entry_threshold

    def generate_signals(self, df):
        df = df.copy()
        df['VWAP'] = compute_vwap(df)
        df['deviation'] = (df['Close'] - df['VWAP']) / df['VWAP'] * 100
        df['signal'] = 0
        df.loc[df['deviation'] > self.entry_threshold, 'signal'] = -1
        df.loc[df['deviation'] < -self.entry_threshold, 'signal'] = 1
        return df

class RSIMeanReversion:
    """RSI均值回归策略"""
    def __init__(self, rsi_period=5, oversold=30, overbought=70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df):
        df = df.copy()
        df['RSI'] = compute_rsi(df['Close'], self.rsi_period)
        df['signal'] = 0
        df.loc[df['RSI'] < self.oversold, 'signal'] = 1
        df.loc[df['RSI'] > self.overbought, 'signal'] = -1
        return df

class MomentumStrategy:
    """趋势动量策略"""
    def __init__(self, lookback=10, threshold=0.005):
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, df):
        df = df.copy()
        df['ret'] = df['Close'].pct_change(self.lookback)
        df['signal'] = 0
        df.loc[df['ret'] > self.threshold, 'signal'] = 1
        df.loc[df['ret'] < -self.threshold, 'signal'] = -1
        return df

class ORBStrategy:
    """ORB突破策略"""
    def __init__(self, orb_minutes=15):
        self.orb_minutes = orb_minutes

    def generate_signals(self, df):
        df = df.copy()
        df = df.reset_index()
        if 'Datetime' in df.columns:
            dt_col = 'Datetime'
        else:
            dt_col = df.columns[0]

        # 统一时区处理（处理index已经是datetime的情况）
        dt_vals = df[dt_col]
        if hasattr(dt_vals, 'dt'):
            dt_series = dt_vals.dt.tz_convert('UTC').dt.tz_localize(None)
        else:
            dt_series = pd.to_datetime(dt_vals, utc=True).dt.tz_localize(None)
        df['hour'] = dt_series.dt.hour
        df['minute'] = dt_series.dt.minute
        df['date'] = dt_series.dt.date
        df = df.set_index(dt_col)

        orb_start = 9 * 60 + 30
        orb_end = orb_start + self.orb_minutes

        df['orb_high'] = np.nan
        df['orb_low'] = np.nan

        for date in df['date'].unique():
            mask = df['date'] == date
            day_data = df[mask]
            time_min = day_data['hour'] * 60 + day_data['minute']
            orb_mask = (time_min >= orb_start) & (time_min < orb_end)
            if orb_mask.any():
                df.loc[mask, 'orb_high'] = day_data.loc[orb_mask, 'High'].max()
                df.loc[mask, 'orb_low'] = day_data.loc[orb_mask, 'Low'].min()

        df['signal'] = 0
        after_orb = df['orb_high'].notna()
        df.loc[after_orb & (df['Close'] > df['orb_high']), 'signal'] = 1
        df.loc[after_orb & (df['Close'] < df['orb_low']), 'signal'] = -1
        return df

class VWAPMomentumCombo:
    """VWAP + 动量组合策略"""
    def __init__(self, entry_threshold=0.01, momentum_threshold=0.002):
        self.entry_threshold = entry_threshold
        self.momentum_threshold = momentum_threshold

    def generate_signals(self, df):
        df = df.copy()
        df['VWAP'] = compute_vwap(df)
        df['deviation'] = (df['Close'] - df['VWAP']) / df['VWAP'] * 100
        df['ret5'] = df['Close'].pct_change(5)

        df['signal'] = 0
        df.loc[(df['deviation'] > self.entry_threshold) & (df['ret5'] > self.momentum_threshold), 'signal'] = 1
        df.loc[(df['deviation'] < -self.entry_threshold) & (df['ret5'] < -self.momentum_threshold), 'signal'] = -1
        return df

# ============ 回测引擎 ============
class Backtester:
    def __init__(self, initial_capital=100000, commission=0.0005, slippage=0.0002):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(self, df, strategy_name, max_hold_bars=25):
        position = 0
        entry_price = 0
        entry_bar = 0
        trades = []
        capital = self.initial_capital

        df = df.reset_index(drop=True)

        for i in range(1, len(df)):
            signal = df['signal'].iloc[i]

            # 入场
            if position == 0 and signal != 0:
                entry_price = df['Close'].iloc[i] * (1 + self.slippage if signal > 0 else 1 - self.slippage)
                capital -= entry_price * self.commission
                position = signal
                entry_bar = i

            # 出场
            elif position != 0:
                bars_held = i - entry_bar
                should_exit = False

                # 反向信号
                if signal == -position:
                    should_exit = True

                # 超时
                if bars_held >= max_hold_bars:
                    should_exit = True

                # RSI回归中性
                if 'RSI' in df.columns:
                    rsi = df['RSI'].iloc[i]
                    if position > 0 and rsi > 50:
                        should_exit = True
                    if position < 0 and rsi < 50:
                        should_exit = True

                if should_exit:
                    exit_price = df['Close'].iloc[i] * (1 - self.slippage if position > 0 else 1 + self.slippage)
                    capital -= exit_price * self.commission
                    pnl = (exit_price - entry_price) * position * 100
                    capital += pnl
                    trades.append({
                        'exit_time': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'Long' if position > 0 else 'Short',
                        'pnl': pnl,
                        'capital': capital
                    })
                    position = 0

        if len(trades) == 0:
            return {
                'strategy': strategy_name,
                'trades': 0, 'win_rate': 0, 'total_pnl': 0,
                'sharpe': 0, 'max_drawdown': 0, 'final_capital': self.initial_capital
            }

        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        returns = pd.Series(pnls) / self.initial_capital
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        cumulative = pd.Series([t['capital'] for t in trades])
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_dd = drawdowns.min()

        return {
            'strategy': strategy_name,
            'trades': len(trades),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean([p for p in pnls if p < 0]) if [p for p in pnls if p < 0] else 0,
            'total_pnl': capital - self.initial_capital,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'final_capital': capital
        }

# ============ 主程序 ============
def main():
    parser = argparse.ArgumentParser(description='SPY日内策略回测')
    parser.add_argument('--data', default='spy_hourly.csv', help='数据文件路径')
    parser.add_argument('--strategy', nargs='+',
                        choices=['vwap', 'rsi', 'momentum', 'orb', 'combo', 'all'],
                        default=['all'], help='策略选择')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    parser.add_argument('--commission', type=float, default=0.0005, help='佣金率')
    parser.add_argument('--slippage', type=float, default=0.0002, help='滑点率')
    parser.add_argument('--max-hold', type=int, default=25, help='最大持仓K线数')
    parser.add_argument('--output', default='backtest_results.csv', help='结果输出文件')

    args = parser.parse_args()

    # 读取数据
    if not os.path.exists(args.data):
        print(f"数据文件不存在: {args.data}")
        return

    print("=" * 60)
    print(f"SPY日内策略回测 v4")
    print("=" * 60)
    print(f"\n数据文件: {args.data}")

    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    print(f"数据条数: {len(df):,}")
    print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")

    # 策略映射
    strategy_map = {
        'vwap': ('VWAP均值回归', VWAPMeanReversion(entry_threshold=0.015), 20),
        'rsi': ('RSI均值回归', RSIMeanReversion(rsi_period=5, oversold=30, overbought=70), 25),
        'momentum': ('趋势动量', MomentumStrategy(lookback=10, threshold=0.005), 20),
        'orb': ('ORB突破', ORBStrategy(orb_minutes=15), 25),
        'combo': ('VWAP+动量组合', VWAPMomentumCombo(entry_threshold=0.01), 25),
    }

    if 'all' in args.strategy:
        strategies = list(strategy_map.values())
    else:
        strategies = [strategy_map[s] for s in args.strategy]

    # 回测
    backtester = Backtester(
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage
    )

    results = []

    for name, strategy, max_hold in strategies:
        print(f"\n{'='*40}")
        print(f"回测: {name}")
        print("="*40)

        try:
            df_signals = strategy.generate_signals(df)
            result = backtester.run(df_signals, name, max_hold_bars=args.max_hold)
            results.append(result)

            print(f"交易次数: {result['trades']}")
            print(f"胜率: {result['win_rate']:.1%}")
            print(f"总盈亏: ${result['total_pnl']:.2f}")
            print(f"夏普比率: {result['sharpe']:.3f}")
            print(f"最大回撤: {result['max_drawdown']:.1%}")
            print(f"最终资金: ${result['final_capital']:.2f}")
        except Exception as e:
            print(f"出错: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    print("\n" + "=" * 60)
    print("策略表现汇总")
    print("=" * 60)
    print(f"{'策略':<20} {'交易次数':>8} {'胜率':>8} {'总盈亏':>12} {'夏普':>8} {'最大回撤':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['strategy']:<18} {r['trades']:>8} {r['win_rate']:>8.1%} {r['total_pnl']:>12.2f} {r['sharpe']:>8.3f} {r['max_drawdown']:>10.1%}")

    # 保存
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"\n结果已保存到 {args.output}")

if __name__ == "__main__":
    main()