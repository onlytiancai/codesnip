# Python framework for intraday short-term signals using pandas
# - Generates synthetic minute-level data if no CSV provided
# - Detects simple K-line patterns: Hammer, Bullish/Bearish Engulfing
# - Detects volume surge and open interest confirmation
# - Combines signals with scoring to generate entry signals
# - Simulates simple backtest with fixed TP/SL or time-based exit
#
# To adapt to real data: replace synthetic data generation with
# df = pd.read_csv('your_minute_data.csv', parse_dates=['datetime']).set_index('datetime')
#
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

def generate_synthetic_data(start='2025-01-01 09:00', minutes=600):
    """Generate synthetic minute OHLCV + open_interest time series for demo."""
    idx = pd.date_range(start=pd.to_datetime(start), periods=minutes, freq='T')
    price = 100 + np.cumsum(np.random.normal(loc=0, scale=0.02, size=minutes))  # small walk
    # add occasional jumps to simulate events
    for j in range(20, minutes, 120):
        price[j:j+3] += np.random.choice([0.5, -0.5])
    df = pd.DataFrame(index=idx)
    df['close'] = price
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.abs(np.random.normal(0, 0.02, size=minutes))
    df['low'] = np.minimum(df['open'], df['close']) - np.abs(np.random.normal(0, 0.02, size=minutes))
    # volume correlated with absolute returns, add bursts
    base_vol = 100 + np.random.poisson(2, size=minutes)
    vol_bursts = (np.abs(df['close'].diff().fillna(0)) > 0.1).astype(int) * 500
    df['volume'] = base_vol + vol_bursts + np.random.poisson(5, size=minutes)
    # open interest random walk with occasional jumps correlated with price jumps
    oi = 1000 + np.cumsum(np.random.normal(0, 1, size=minutes))
    oi[20::120] += np.random.choice([50, -50], size=len(oi[20::120]))
    df['open_interest'] = oi
    return df

# Indicator helpers
def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def atr(df, window=14):
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1)
    tr = tr.max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()

# Pattern detectors
def detect_hammer(df):
    # Hammer: lower shadow at least 2x body, small upper shadow
    open_ = df['open']; close = df['close']; high = df['high']; low = df['low']
    body = (close - open_).abs()
    lower_shadow = np.where(close >= open_, open_ - low, close - low)
    upper_shadow = np.where(close >= open_, high - close, high - open_)
    cond = (lower_shadow >= 2 * body) & (upper_shadow <= 0.5 * body) & (body > 0)
    return cond

def detect_shooting_star(df):
    # Shooting star: opposite of hammer (long upper shadow)
    open_ = df['open']; close = df['close']; high = df['high']; low = df['low']
    body = (close - open_).abs()
    upper_shadow = np.where(close >= open_, high - close, high - open_)
    lower_shadow = np.where(close >= open_, open_ - low, close - low)
    cond = (upper_shadow >= 2 * body) & (lower_shadow <= 0.5 * body) & (body > 0)
    return cond

def detect_bullish_engulfing(df):
    # Previous candle bearish, current candle bullish and engulfs previous body
    prev_open = df['open'].shift(1); prev_close = df['close'].shift(1)
    cond_prev_bear = prev_close < prev_open
    cond_curr_bull = df['close'] > df['open']
    cond_engulf = (df['close'] > prev_open) & (df['open'] < prev_close)
    return cond_prev_bear & cond_curr_bull & cond_engulf

def detect_bearish_engulfing(df):
    prev_open = df['open'].shift(1); prev_close = df['close'].shift(1)
    cond_prev_bull = prev_close > prev_open
    cond_curr_bear = df['close'] < df['open']
    cond_engulf = (df['open'] > prev_close) & (df['close'] < prev_open)
    return cond_prev_bull & cond_curr_bear & cond_engulf

# Volume surge and OI confirmation
def detect_volume_surge(df, short=5, threshold=1.8):
    vol_sma = sma(df['volume'], short)
    return (df['volume'] > vol_sma * threshold)

def detect_oi_confirmation(df):
    # d_oi * d_price > 0 indicates new money in direction
    d_oi = df['open_interest'].diff().fillna(0)
    d_price = df['close'].diff().fillna(0)
    return (d_oi * d_price) > 0

# Trend via MA (multi-timeframe approximated by different windows)
def compute_trend(df, ma_short=5, ma_long=20):
    ma_s = sma(df['close'], ma_short)
    ma_l = sma(df['close'], ma_long)
    trend_up = ma_s > ma_l
    trend_down = ma_s < ma_l
    return trend_up, trend_down, ma_s, ma_l

# Combine signals into score and final signal
def generate_signals(df):
    df = df.copy()
    df['hammer'] = detect_hammer(df)
    df['shooting_star'] = detect_shooting_star(df)
    df['bull_engulf'] = detect_bullish_engulfing(df)
    df['bear_engulf'] = detect_bearish_engulfing(df)
    df['vol_surge'] = detect_volume_surge(df, short=5, threshold=1.8)
    df['oi_conf'] = detect_oi_confirmation(df)
    df['trend_up'], df['trend_down'], df['ma_s'], df['ma_l'] = compute_trend(df)

    # scoring
    df['score_long'] = 0
    df['score_short'] = 0

    # pattern scores (give higher weight to engulfing)
    df.loc[df['hammer'], 'score_long'] += 2
    df.loc[df['bull_engulf'], 'score_long'] += 3
    df.loc[df['shooting_star'], 'score_short'] += 2
    df.loc[df['bear_engulf'], 'score_short'] += 3

    # volume and OI confirmation
    df.loc[df['vol_surge'] & df['oi_conf'], 'score_long'] += 2  # if both true, strong
    df.loc[df['vol_surge'] & df['oi_conf'], 'score_short'] += 2

    # trend alignment: if trend_up then +1 for long, if trend_down then +1 for short
    df.loc[df['trend_up'], 'score_long'] += 1
    df.loc[df['trend_down'], 'score_short'] += 1

    # thresholding into signals: require score >= 4 to open (tunable)
    df['long_signal'] = df['score_long'] >= 4
    df['short_signal'] = df['score_short'] >= 4

    # avoid both signals true (rare) by cancelling
    df.loc[df['long_signal'] & df['short_signal'], ['long_signal', 'short_signal']] = False

    return df

# Simple backtest engine
def backtest(df, tp_pct=0.002, sl_pct=0.001, max_holding_minutes=60, init_cash=100000, position_pct=0.1):
    """
    - tp_pct: take profit as fraction of entry price (e.g., 0.002 = 0.2%)
    - sl_pct: stop loss fraction
    - max_holding_minutes: time-based exit
    - position_pct: fraction of portfolio to allocate per trade
    """
    df = df.copy()
    trades = []
    cash = init_cash
    position = 0.0  # positive for long, negative for short
    entry_price = None
    entry_time = None
    entry_size = 0.0

    for t, row in df.iterrows():
        price = row['close']
        # Check existing position for exit conditions
        if position != 0:
            ret = (price - entry_price) / entry_price if position > 0 else (entry_price - price) / entry_price
            # TP/SL
            if ret >= tp_pct or ret <= -sl_pct:
                # close position
                cash += position * price  # realize pnl (position is signed number of units)
                trades.append({'entry_time': entry_time, 'exit_time': t, 'entry_price': entry_price,
                               'exit_price': price, 'position': position, 'pnl': position * (price - entry_price)})
                position = 0.0
                entry_price = None
                entry_time = None
                entry_size = 0.0
                continue
            # time based exit
            if (t - entry_time) >= pd.Timedelta(minutes=max_holding_minutes):
                cash += position * price
                trades.append({'entry_time': entry_time, 'exit_time': t, 'entry_price': entry_price,
                               'exit_price': price, 'position': position, 'pnl': position * (price - entry_price)})
                position = 0.0
                entry_price = None
                entry_time = None
                entry_size = 0.0
                continue

        # If no open position, check signals
        if position == 0:
            if row['long_signal']:
                allocation = cash * position_pct
                entry_size = allocation / price  # number of contracts/shares
                position = entry_size
                entry_price = price
                entry_time = t
                cash -= allocation  # reserve cash
            elif row['short_signal']:
                allocation = cash * position_pct
                entry_size = allocation / price
                position = -entry_size
                entry_price = price
                entry_time = t
                cash -= allocation

    # close any open position at last price
    if position != 0:
        last_price = df['close'].iloc[-1]
        cash += position * last_price
        trades.append({'entry_time': entry_time, 'exit_time': df.index[-1], 'entry_price': entry_price,
                       'exit_price': last_price, 'position': position, 'pnl': position * (last_price - entry_price)})

    # summarize
    trades_df = pd.DataFrame(trades)
    total_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0.0
    num_trades = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean() if num_trades > 0 else np.nan
    avg_pnl = trades_df['pnl'].mean() if num_trades > 0 else np.nan
    return {'final_cash': cash, 'total_pnl': total_pnl, 'num_trades': num_trades, 'win_rate': win_rate, 'avg_pnl': avg_pnl, 'trades': trades_df}

# Run demo
if __name__ == '__main__':
    df = generate_synthetic_data(minutes=720)  # 12 hours of minute data
    df_signals = generate_signals(df)
    result = backtest(df_signals, tp_pct=0.002, sl_pct=0.001, max_holding_minutes=120, init_cash=100000, position_pct=0.05)

    # show some outputs
    print("=== Sample Signals (first 50 rows) ===")
    print(df_signals[['close','volume','open_interest','hammer','bull_engulf','vol_surge','oi_conf','score_long','score_short','long_signal','short_signal']].head(50))
    print("\n=== Backtest Summary ===")
    for k,v in result.items():
        if k != 'trades':
            print(f"{k}: {v}")
    print("\n=== Trades ===")
    print(result['trades'].head(20))

    # Save trades to CSV for inspection
    result['trades'].to_csv('trades_demo.csv', index=False)
    print("\nTrades saved to trades_demo.csv")
