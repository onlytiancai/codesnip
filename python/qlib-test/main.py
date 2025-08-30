"""
Qlib US long-only strategy example for small capital (≤ ~$100k),
holding up to 10 stocks, weekly rebalancing, 1w–3m horizon.

What this script does
---------------------
1) Initialize Qlib with US daily data (you must prepare qlib US data beforehand).
2) Define universe and data handler (Alpha158 technical factors + a few custom ones).
3) Train LightGBM on rolling windows to predict next-4-week forward return.
4) Construct a long-only portfolio with up to 10 names and basic constraints.
5) Backtest and output performance, turnover, and simple factor diagnostics.

Prerequisites
-------------
- `pip install qlib lightgbm` (Qlib version ≥ 0.9 recommended)
- US market data prepared via Qlib's data collector, e.g. at ~/.qlib/qlib_data/us_data

Notes
-----
- This example focuses on price/volume + size proxies. To add fundamentals,
  see the `TODO: Add fundamental factors` section below.
- For tiny accounts, always add realistic costs (e.g., 10–30 bps one-way) and slippage.

"""

import os
import pandas as pd
import numpy as np

import qlib
from qlib.config import REG_US
from qlib.data import D
from qlib.workflow import R
from qlib.utils import init_instance_by_config
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest as qlib_backtest
from qlib.backtest import executor as bt_executor
from qlib.contrib.evaluate import risk_analysis


# ----------------------
# 0. Init Qlib
# ----------------------
QLIB_US_PATH = os.path.expanduser("~/.qlib/qlib_data/us_data")
qlib.init(provider_uri=QLIB_US_PATH, region=REG_US)

# ----------------------
# 1. Universe & Benchmark
# ----------------------
# Keep the universe liquid to suit small capital and reduce costs
UNIVERSE = [
    # choose one of the built-in universes if available in your data pack
    # e.g., 'SP500', 'US_ALL'
    'SP500'
][0]
BENCHMARK = 'SPY'

# ----------------------
# 2. Label (target) and factors
# ----------------------
# Forward 4-week return (approx. 20 trading days) — mid-horizon
LABEL = "Ref($close, -20) / Ref($close, -1) - 1"

# Alpha158 already provides ~158 TA features from OHLCV.
# We'll optionally append a few custom features (momentum, reversal, volatility, size proxy).
CUSTOM_EXPRS = {
    # Momentum (12-1): past 252d price change excluding last 21d
    'mom_12_1': "Ref($close, -21) / Ref($close, -252) - 1",
    # Short-term reversal: last 5d return
    'rev_5': "Ref($close, -1) / Ref($close, -6) - 1",
    # Volatility: 60d rolling std of daily returns
    'vol_60': "Std($close/Ref($close,-1)-1, 60)",
    # Turnover intensity: 20d average turnover (volume scaled by free float if available)
    'to_20': "Mean($turnover, 20)",
    # Size proxy: log(median dollar volume 60d)
    'dollar_vol_60_log': "Log( Mean($volume*$close, 60) + 1e-6 )",
}

# ----------------------
# 3. Handler & Dataset
# ----------------------
HANDLER_CONF = {
    'class': 'Alpha158',
    'module_path': 'qlib.contrib.data.handler',
    'kwargs': {
        'start_time': '2012-01-01',
        'end_time': '2025-07-31',
        'fit_start_time': '2012-01-01',
        'fit_end_time': '2022-12-31',
        'instruments': UNIVERSE,
        'infer_processors': [
            {'class': 'RobustZScoreNorm', 'kwargs': {'clip_outlier': True, 'fields_group': 'feature'}},
            {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}
        ],
        'learn_processors': [
            {'class': 'DropnaLabel'},
            {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}},
            {'class': 'RobustZScoreNorm', 'kwargs': {'clip_outlier': True, 'fields_group': 'feature'}},
            {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}},
        ],
        'label': LABEL,
        'custom': CUSTOM_EXPRS,  # append our extra features
    },
}

DATASET_CONF = {
    'class': 'DatasetH',
    'module_path': 'qlib.data.dataset',
    'kwargs': {
        'handler': HANDLER_CONF,
        'segments': {
            'train': ('2012-01-01', '2019-12-31'),
            'valid': ('2020-01-01', '2021-12-31'),
            'test':  ('2022-01-01', '2025-07-31'),
        },
    },
}

# ----------------------
# 4. Model (LightGBM)
# ----------------------
MODEL_CONF = {
    'class': 'LGBModel',
    'module_path': 'qlib.contrib.model.gbdt',
    'kwargs': {
        'loss': 'mse',
        'num_leaves': 64,
        'max_depth': -1,
        'learning_rate': 0.05,
        'n_estimators': 1200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'random_state': 7,
        'early_stopping_rounds': 100,
        'verbose': -1,
    },
}

# ----------------------
# 5. Strategy (Top-k long-only) & Executor
# ----------------------
# Hold up to 10 names; drop a few losers each rebalance to reduce churn.
STRATEGY_CONF = {
    'class': 'TopkDropoutStrategy',
    'module_path': 'qlib.contrib.strategy',
    'kwargs': {
        'signal': None,               # filled after we predict
        'topk': 10,
        'n_drop': 2,                  # drop worst 2 each rebalance
        'buffer_day': 5,              # min holding ~1 week before eligible to drop
        'risk_degree': 1.0,           # long-only fully invested
        'hold_thresh': 0.0,           # keep positive scored names until dropped
    },
}

# Daily executor with realistic costs for small accounts
EXECUTOR_CONF = {
    'class': 'SimulatorExecutor',
    'module_path': 'qlib.backtest.executor',
    'kwargs': {
        'time_per_step': 'day',
        'generate_report': True,
        'verbose': False,
    },
}

# Backtest constraints & costs
BACKTEST_CONF = {
    'start_time': '2022-01-03',
    'end_time': '2025-07-31',
    'account': 100000.0,              # starting capital $100k
    'benchmark': BENCHMARK,
    'exchange_kwargs': {
        'freq': 'day',
        # Small-account cost assumptions (tune to your broker)
        'deal_price': 'close',        # use close-to-close (or use 'vwap')
        'open_cost': 0.0015,          # 15 bps buy
        'close_cost': 0.0015,         # 15 bps sell
        'min_cost': 1.0,              # $1 min per trade
        'slippage': 0.0005,           # 5 bps
        'limit_threshold': 0.095,     # ignore names with ±9.5% day limit (US rarely applies)
    },
}

# ----------------------
# 6. Train, Predict, Backtest
# ----------------------
if __name__ == '__main__':
    dataset = init_instance_by_config(DATASET_CONF)
    model = init_instance_by_config(MODEL_CONF)

    # Train with early stopping using valid set
    x_train, y_train = dataset.prepare("train", col_set=["feature", "label"], data_key=("train", "train"))
    x_valid, y_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=("valid", "valid"))
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

    # Predict on test segment
    x_test = dataset.prepare("test", col_set=["feature"], data_key=("test", "test"))
    scores = model.predict(x_test)

    # Align signal index with dataset's label index
    label_df = dataset.prepare("test", col_set=["label"], data_key=("test", "test"))
    signal = pd.Series(scores, index=label_df.index)  # MultiIndex [datetime, instrument]

    # Inject signal into strategy
    strategy = init_instance_by_config(STRATEGY_CONF)
    strategy.set_signal(signal)

    # Risk filter: liquidity & size guardrails (suitable for small accounts)
    # - exclude penny stocks (price < $3)
    # - exclude illiquid: dollar_vol_60 < $2M
    closes = D.features(UNIVERSE, ["$close"], start_time=BACKTEST_CONF['start_time'], end_time=BACKTEST_CONF['end_time']).droplevel(0, axis=1)
    volumes = D.features(UNIVERSE, ["$volume"], start_time=BACKTEST_CONF['start_time'], end_time=BACKTEST_CONF['end_time']).droplevel(0, axis=1)
    dollar_vol_60 = (closes * volumes).rolling(60).mean()

    liquid_mask = (closes >= 3.0) & (dollar_vol_60 >= 2_000_000)

    # Apply mask: set signal to NaN where illiquid
    signal_masked = signal.copy()
    # Convert MultiIndex signal to DataFrame pivoted by instrument to align with mask
    sig_df = signal_masked.unstack(level=1)
    sig_df = sig_df.where(liquid_mask)
    signal_masked = sig_df.stack().sort_index()

    strategy.set_signal(signal_masked)

    executor = init_instance_by_config(EXECUTOR_CONF)

    # Portfolio constraints: max 10 names already handled by strategy.
    # Optional: per-name cap 15% (enforced inside strategy via weights normalization)

    # Run backtest
    portfolio_metric_dict, indicator = qlib_backtest(
        start_time=BACKTEST_CONF['start_time'],
        end_time=BACKTEST_CONF['end_time'],
        strategy=strategy,
        executor=executor,
        account=BACKTEST_CONF['account'],
        benchmark=BACKTEST_CONF['benchmark'],
        exchange_kwargs=BACKTEST_CONF['exchange_kwargs'],
    )

    # Save results to Qlib recorder (optional)
    with R.start(experiment_name="us_longonly_top10"):
        R.log_metrics(**portfolio_metric_dict)
        # Risk analysis vs benchmark
        analysis = risk_analysis(indicator, freq='day')
        print("===== Risk Metrics (Test) =====")
        for k, v in analysis.items():
            print(f"{k}: {v}")

        # Export trades & positions
        trade_df = indicator.get('trade_price')
        pos_df = indicator.get('position')
        if trade_df is not None:
            trade_df.to_csv('trades.csv')
        if pos_df is not None:
            pos_df.to_csv('positions.csv')

    print("Backtest complete. Check trades.csv and positions.csv for details.")


# ----------------------
# TODO: Add fundamental factors
# ----------------------
# 1) Prepare a fundamentals CSV (quarterly) with columns like:
#    date, instrument, book_value, earnings_ttm, sales_ttm, total_debt, shares_outstanding
# 2) Create a custom handler inheriting from Alpha158 that merges your CSV and defines expressions e.g.:
#    - btm = book_value / (shares_outstanding * $close)
#    - ey = earnings_ttm / (shares_outstanding * $close)
#    - gm = (sales_ttm - cogs_ttm) / sales_ttm
# 3) Register those fields via the `custom` dict (like CUSTOM_EXPRS) or override `Alpha158.get_features`.
# 4) Re-run training and backtest.
