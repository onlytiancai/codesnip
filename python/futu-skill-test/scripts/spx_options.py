#!/usr/bin/env python3
"""
SPX Options Portfolio Display

Uses yfinance to get market data (SPX, VIX, SPY, ES futures),
and futu OpenAPI to get SPX option positions with real Greeks.
Uses futu get_stock_quote to fetch IV, Delta, Gamma, Vega, Theta, Rho, Open Interest.

Usage:
    python scripts/spx_options.py
"""

import sys
import argparse
import json
import time
from datetime import datetime


def log(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


try:
    import yfinance as yf
except ImportError:
    log("ERROR: install yfinance: pip install yfinance")
    sys.exit(1)

try:
    from futu import *
except ImportError:
    log("ERROR: install futu-api: pip install futu-api")
    sys.exit(1)


# ============== Market Data ==============

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def get_market_data(symbols: list = None) -> dict:
    """Get market data via yfinance with retry logic."""
    if symbols is None:
        symbols = ['SPX', 'VIX', 'SPY']

    index_map = {
        'SPX': '^SPX',
        '^SPX': '^SPX',
        'VIX': '^VIX',
        '^VIX': '^VIX',
        'SPY': 'SPY',
        'ES': 'ES=F',
    }

    result = {}
    failed_symbols = []

    for symbol in symbols:
        ticker_sym = index_map.get(symbol, symbol)
        log(f"Fetching market data for {symbol} ({ticker_sym})...")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                ticker = yf.Ticker(ticker_sym)
                info = ticker.info

                if info is None or info == {}:
                    raise ValueError("Empty response from yfinance")

                result[symbol] = {
                    'symbol': symbol,
                    'price': info.get('regularMarketPrice'),
                    'change': info.get('regularMarketChange'),
                    'change_percent': info.get('regularMarketChangePercent'),
                    'open': info.get('regularMarketOpen'),
                    'previous_close': info.get('regularMarketPreviousClose'),
                    'day_high': info.get('dayHigh'),
                    'day_low': info.get('dayLow'),
                    'volume': info.get('volume'),
                    '52_week_high': info.get('fiftyTwoWeekHigh'),
                    '52_week_low': info.get('fiftyTwoWeekLow'),
                }
                log(f"Successfully fetched {symbol}: price={result[symbol]['price']}")
                break

            except Exception as e:
                error_msg = str(e)
                log(f"WARNING: Attempt {attempt}/{MAX_RETRIES} failed for {symbol}: {error_msg}")

                if attempt < MAX_RETRIES:
                    log(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    log(f"ERROR: All {MAX_RETRIES} attempts failed for {symbol}")
                    failed_symbols.append(symbol)
                    result[symbol] = {'symbol': symbol, 'error': error_msg}

        time.sleep(1)  # Delay between different symbols

    if failed_symbols:
        log(f"ERROR: Failed to fetch market data for: {', '.join(failed_symbols)}")
        log("ERROR: Cannot proceed without market data. Exiting.")
        sys.exit(1)

    return result


# ============== Option Quote via Futu API ==============

def get_option_quotes(codes: list) -> dict:
    """Fetch option quote data (IV, Greeks, OI, etc.) from Futu API."""
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

    # Subscribe first
    ret, _ = quote_ctx.subscribe(codes, [SubType.QUOTE])
    if ret != RET_OK:
        quote_ctx.close()
        return {}

    ret, data = quote_ctx.get_stock_quote(codes)
    quote_ctx.close()

    if ret != RET_OK:
        return {}

    # Index by code
    result = {}
    for _, row in data.iterrows():
        code = row['code']
        result[code] = {
            'iv': row.get('implied_volatility'),
            'delta': row.get('delta'),
            'gamma': row.get('gamma'),
            'vega': row.get('vega'),
            'theta': row.get('theta'),
            'rho': row.get('rho'),
            'open_interest': row.get('open_interest'),
            'volume': row.get('volume'),
            'premium': row.get('premium'),
            'contract_size': row.get('contract_size'),
            'last_price': row.get('last_price'),
            'high_price': row.get('high_price'),
            'low_price': row.get('low_price'),
            'prev_close_price': row.get('prev_close_price'),
            'turnover': row.get('turnover'),
        }
    return result


# ============== Portfolio ==============

def get_positions(trd_env: str = 'REAL'):
    """Get portfolio positions via futu OpenAPI"""
    try:
        log("Connecting to Futu OpenAPI to fetch positions...")
        trd_ctx = OpenSecTradeContext(host='127.0.0.1', port=11111)
        trd_env_enum = TrdEnv.REAL if trd_env == 'REAL' else TrdEnv.SIMULATE
        ret, data = trd_ctx.position_list_query(trd_env=trd_env_enum)
        trd_ctx.close()

        if ret == RET_OK:
            log(f"Successfully fetched positions, count: {len(data)}")
            return data
        log(f"ERROR: Failed to get positions: {data}")
        return None
    except Exception as e:
        log(f"ERROR: Error connecting to Futu OpenAPI: {e}")
        return None


def parse_option_code(code: str) -> dict:
    """Parse SPX option code like US.SPX260417P6500000"""
    if 'SPX' not in code:
        return None

    parts = code.split('SPX')
    if len(parts) < 2:
        return None

    rest = parts[1]
    if rest.startswith('.'):
        rest = rest[1:]

    if len(rest) < 7:
        return None

    date_str = rest[:6]
    type_char = rest[6]
    strike_cents = rest[7:]

    try:
        expiry = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
        strike = float(strike_cents) / 1000
        return {
            'expiry': expiry,
            'type': 'call' if type_char == 'C' else 'put',
            'strike': strike,
        }
    except:
        return None


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description='SPX Options Portfolio Display')
    parser.add_argument('--trd-env', default='REAL', choices=['REAL', 'SIMULATE'])
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    log("Starting SPX Options Portfolio script")
    log(f"Trading environment: {args.trd_env}")

    # Get market data
    log("Fetching market data for SPX, VIX, SPY, ES...")
    market = get_market_data(['SPX', 'VIX', 'SPY', 'ES'])
    log("Market data fetched successfully")

    # Get positions
    positions = get_positions(args.trd_env)
    if positions is None or positions.empty:
        log("ERROR: No positions or failed to fetch positions")
        return

    # Filter SPX options
    spx_opts = positions[positions['code'].str.contains('SPX', case=False)].copy()
    log(f"Found {len(spx_opts)} SPX option positions")

    if spx_opts.empty:
        log("WARNING: No SPX options positions found")
        return

    # Collect all codes for batch quote fetch
    codes = spx_opts['code'].tolist()

    # Fetch option quotes (IV, Greeks, OI, volume, etc.) from Futu API
    log("Fetching option quotes from Futu API...")
    quotes = get_option_quotes(codes)
    log(f"Fetched quotes for {len(quotes)} options")

    # Parse and enrich positions
    rows = []
    for _, pos in spx_opts.iterrows():
        code = pos['code']
        parsed = parse_option_code(code)

        if not parsed:
            continue

        expiry = parsed['expiry']
        opt_type = parsed['type']
        strike = parsed['strike']

        # Days to expiry
        try:
            exp_date = datetime.strptime(expiry, '%Y-%m-%d')
            days_to_expiry = (exp_date - datetime.now()).days
        except:
            days_to_expiry = 0

        # Cost info from position
        qty = pos['qty']
        cost_price = pos['cost_price']
        market_val = pos['market_val']
        pl_val = pos.get('pl_val', 0)

        # Derive current price from market_val
        if qty != 0 and market_val != 0:
            current_price = market_val / (qty * 100)
        else:
            current_price = None

        # Get quote data from API
        q = quotes.get(code, {})

        rows.append({
            'code': code,
            'expiry': expiry,
            'days_to_expiry': days_to_expiry,
            'type': opt_type.upper(),
            'strike': strike,
            'qty': qty,
            'cost_price': round(cost_price, 2),
            'current_price': round(current_price, 2) if current_price else None,
            'market_value': round(market_val, 2),
            'pl_unrealized': round(pl_val, 2),
            'iv': q.get('iv'),
            'delta': q.get('delta'),
            'gamma': q.get('gamma'),
            'vega': q.get('vega'),
            'theta': q.get('theta'),
            'rho': q.get('rho'),
            'open_interest': q.get('open_interest'),
            'volume': q.get('volume'),
            'premium': q.get('premium'),
            'high_price': q.get('high_price'),
            'low_price': q.get('low_price'),
            'prev_close_price': q.get('prev_close_price'),
        })

    if args.json:
        output = {
            'market': market,
            'positions': rows,
            'calculated_at': datetime.now().isoformat(),
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        # ---- Market Data ----
        print("=" * 110)
        print("MARKET DATA")
        print("=" * 110)
        for sym, data in market.items():
            if 'error' not in data:
                print(f"\n{sym}")
                print(f"  Price:         {data.get('price', 'N/A')}")
                print(f"  Change:        {data.get('change', 'N/A')} ({data.get('change_percent', 'N/A')}%)")
                print(f"  Day Range:     {data.get('day_low', 'N/A')} - {data.get('day_high', 'N/A')}")
                print(f"  52W Range:    {data.get('52_week_low', 'N/A')} - {data.get('52_week_high', 'N/A')}")
            else:
                print(f"\n{sym}: Failed - {data.get('error')}")

        # ---- Position Table ----
        print("\n" + "=" * 110)
        print("SPX OPTIONS")
        print("=" * 110)
        hdr = f"{'Code':<25} {'Expiry':<12} {'DTE':>4} {'Type':<5} {'Strike':>8} {'Qty':>5} {'Cost':>8} {'Current':>8} {'MktVal':>11} {'P/L':>11} {'IV':>7}"
        print(hdr)
        print("-" * 110)

        for r in rows:
            iv_str = f"{r['iv']:.2f}%" if r['iv'] is not None else "N/A"
            curr_str = f"{r['current_price']:.2f}" if r['current_price'] is not None else "N/A"
            print(f"{r['code']:<25} {r['expiry']:<12} {r['days_to_expiry']:>4} {r['type']:<5} {r['strike']:>8.0f} {r['qty']:>5.0f} {r['cost_price']:>8.2f} {curr_str:>8} {r['market_value']:>11.2f} {r['pl_unrealized']:>11.2f} {iv_str:>7}")

        # ---- Greeks ----
        print("\n" + "=" * 110)
        print("GREEKS  (from Futu API)")
        print("=" * 110)
        hdr2 = f"{'Code':<25} {'Delta':>8} {'Gamma':>8} {'Theta':>10} {'Vega':>8} {'Rho':>10}"
        print(hdr2)
        print("-" * 110)
        for r in rows:
            d  = f"{r['delta']:>8.4f}" if r['delta'] is not None else "N/A"
            g  = f"{r['gamma']:>8.4f}" if r['gamma'] is not None else "N/A"
            th = f"{r['theta']:>10.4f}" if r['theta'] is not None else "N/A"
            v  = f"{r['vega']:>8.4f}" if r['vega'] is not None else "N/A"
            rh = f"{r['rho']:>10.4f}" if r['rho'] is not None else "N/A"
            print(f"{r['code']:<25} {d} {g} {th} {v} {rh}")

        # ---- Open Interest & Volume ----
        print("\n" + "=" * 110)
        print("OPEN INTEREST & VOLUME  (from Futu API)")
        print("=" * 110)
        hdr3 = f"{'Code':<25} {'Open Interest':>14} {'Volume':>10} {'Premium':>12} {'High':>10} {'Low':>10} {'Prev Close':>12}"
        print(hdr3)
        print("-" * 110)
        for r in rows:
            oi = f"{r['open_interest']:>14,.0f}" if r['open_interest'] is not None else "N/A"
            vol = f"{r['volume']:>10,.0f}" if r['volume'] is not None else "N/A"
            prem = f"{r['premium']:>12.2f}" if r['premium'] is not None else "N/A"
            hi = f"{r['high_price']:>10.2f}" if r['high_price'] is not None else "N/A"
            lo = f"{r['low_price']:>10.2f}" if r['low_price'] is not None else "N/A"
            pc = f"{r['prev_close_price']:>12.2f}" if r['prev_close_price'] is not None else "N/A"
            print(f"{r['code']:<25} {oi} {vol} {prem} {hi} {lo} {pc}")

        print(f"\nCalculated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
