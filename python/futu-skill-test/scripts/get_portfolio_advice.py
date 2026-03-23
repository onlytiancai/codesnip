#!/usr/bin/env python3
"""
获取投资组合建议 - 统一脚本

整合 SPX/VIX 市场数据和 SPX 期权持仓分析，提供综合投资建议。

功能:
1. 通过 yfinance 获取 SPX 和 VIX 当前价格
2. 通过 futu OpenAPI 获取 SPX 期权持仓
3. 识别策略 (铁鹰、垂直价差等)
4. 计算盈亏和风险
5. 生成投资建议

用法:
    python scripts/get_portfolio_advice.py [--trd-env REAL|SIMULATE] [--json]
"""

import sys
import json
import argparse
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    print("Error: please install yfinance: pip install yfinance")
    sys.exit(1)

try:
    from futu import *
except ImportError:
    print("Error: please install futu-api: pip install futu-api")
    sys.exit(1)


# ============== Market Data Functions ==============

def get_index_info(symbol: str) -> dict:
    """Get index information via yfinance"""
    index_map = {
        'SPX': '^SPX',
        '^SPX': '^SPX',
        'VIX': '^VIX',
        '^VIX': '^VIX',
        'NDX': '^NDX',
        '^NDX': '^NDX',
        'DJI': '^DJI',
        '^DJI': '^DJI',
    }

    ticker = index_map.get(symbol, symbol)
    try:
        idx = yf.Ticker(ticker)
        info = idx.info

        return {
            'symbol': symbol,
            'name': info.get('longName', info.get('shortName', 'N/A')),
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
    except Exception as e:
        return {
            'symbol': symbol,
            'error': str(e)
        }


def get_market_data(symbols: list = None) -> dict:
    """Get market data for multiple symbols"""
    if symbols is None:
        symbols = ['SPX', 'VIX']

    result = {}
    for symbol in symbols:
        info = get_index_info(symbol)
        result[symbol] = info

    return result


# ============== Portfolio Functions ==============

def get_portfolio_positions(trd_env: str = 'REAL'):
    """Get portfolio positions"""
    try:
        trd_ctx = OpenSecTradeContext(host='127.0.0.1', port=11111)
        trd_env_enum = TrdEnv.REAL if trd_env == 'REAL' else TrdEnv.SIMULATE
        ret, data = trd_ctx.position_list_query(trd_env=trd_env_enum)
        trd_ctx.close()

        if ret == RET_OK:
            return data
        else:
            print(f"Failed to get positions: {data}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def identify_strategy(positions: list) -> dict:
    """
    Identify options strategy types

    Supported strategies:
    - Iron Condor: Short OTM Put + Long OTM Put + Short OTM Call + Long OTM Call
    - Put Vertical Spread: Long Put + Short Put (same expiry)
    - Call Vertical Spread: Short Call + Long Call (same expiry)
    - Straddle: Long/Short Call + Long/Short Put (same strike)
    - Strangle: Long/Short Call + Long/Short Put (different strikes)
    """
    strategies = {}

    # Group by expiry
    by_expiry = {}
    for _, pos in positions.iterrows():
        code = pos['code']
        # Parse options code: US.SPX260417P6500000
        if code.startswith('US.SPX') or code.startswith('US..SPX'):
            rest = code.split('SPX')[1] if 'SPX' in code else ''
            if len(rest) < 7:
                continue

            date_str = rest[:6]
            type_char = rest[6] if len(rest) > 6 else None
            strike = float(rest[7:]) / 1000 if len(rest) > 7 else None

            if date_str and type_char and strike:
                expiry = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"

                if expiry not in by_expiry:
                    by_expiry[expiry] = {'puts': [], 'calls': []}

                leg_data = {
                    'code': code,
                    'qty': pos['qty'],
                    'cost_price': pos['cost_price'],
                    'market_val': pos['market_val'],
                    'pl_val': pos.get('pl_val', 0),
                    'strike': strike,
                }

                if type_char == 'P':
                    by_expiry[expiry]['puts'].append(leg_data)
                elif type_char == 'C':
                    by_expiry[expiry]['calls'].append(leg_data)

    # Analyze each expiry
    for expiry, legs in by_expiry.items():
        strategy_name = []
        details = {
            'expiry': expiry,
            'puts': legs['puts'],
            'calls': legs['calls'],
            'total_pl': 0,
        }

        # Check Put side
        if legs['puts']:
            long_puts = [p for p in legs['puts'] if p['qty'] > 0]
            short_puts = [p for p in legs['puts'] if p['qty'] < 0]

            if long_puts and short_puts:
                long_strike = long_puts[0]['strike']
                short_strike = short_puts[0]['strike']
                if short_strike > long_strike:
                    strategy_name.append(f"Bull Put Spread {long_strike}/{short_strike}")
                else:
                    strategy_name.append(f"Bear Put Spread {short_strike}/{long_strike}")
                details['put_spread'] = {
                    'long_strike': long_strike if long_puts else None,
                    'short_strike': short_strike if short_puts else None,
                    'long_qty': sum(p['qty'] for p in long_puts),
                    'short_qty': sum(p['qty'] for p in short_puts),
                }

        # Check Call side
        if legs['calls']:
            long_calls = [c for c in legs['calls'] if c['qty'] > 0]
            short_calls = [c for c in legs['calls'] if c['qty'] < 0]

            if long_calls and short_calls:
                long_strike = long_calls[0]['strike']
                short_strike = short_calls[0]['strike']
                if short_strike > long_strike:
                    strategy_name.append(f"Bear Call Spread {long_strike}/{short_strike}")
                else:
                    strategy_name.append(f"Bull Call Spread {short_strike}/{long_strike}")
                details['call_spread'] = {
                    'long_strike': long_strike if long_calls else None,
                    'short_strike': short_strike if short_calls else None,
                    'long_qty': sum(c['qty'] for c in long_calls),
                    'short_qty': sum(c['qty'] for c in short_calls),
                }

        # Check for Iron Condor
        if legs['puts'] and legs['calls']:
            long_puts = [p for p in legs['puts'] if p['qty'] > 0]
            short_puts = [p for p in legs['puts'] if p['qty'] < 0]
            long_calls = [c for c in legs['calls'] if c['qty'] > 0]
            short_calls = [c for c in legs['calls'] if c['qty'] < 0]

            if all([long_puts, short_puts, long_calls, short_calls]):
                strategy_name.insert(0, "Iron Condor")
                details['is_iron_condor'] = True

        # Calculate total P&L - use futu's pl_val field (more accurate)
        for leg in legs['puts'] + legs['calls']:
            details['total_pl'] += leg.get('pl_val', leg['market_val'] - (leg['cost_price'] * abs(leg['qty']) * 100))

        if strategy_name:
            details['name'] = ' + '.join(strategy_name)
            strategies[expiry] = details

    return strategies


# ============== Analysis Functions ==============

def analyze_iron_condor(strategy: dict, spx_price: float) -> dict:
    """Analyze Iron Condor P&L and risk"""
    result = {
        'spx_price': spx_price,
        'lower_be': None,
        'upper_be': None,
        'max_profit': None,
        'max_loss': None,
        'risk_level': 'unknown',
    }

    if 'put_spread' not in strategy or 'call_spread' not in strategy:
        return result

    put_spread = strategy['put_spread']
    call_spread = strategy['call_spread']

    # Calculate net credit from actual position cost prices
    # Sum P&L for all legs to get total credits received
    total_credit = 0
    for leg in strategy.get('puts', []) + strategy.get('calls', []):
        # For short positions: cost_price is credit received (positive)
        # For long positions: cost_price is debit paid (negative contribution)
        qty = leg.get('qty', 0)
        cost = leg.get('cost_price', 0)
        # Short positions (qty < 0): we received premium
        # Long positions (qty > 0): we paid premium
        if qty < 0:
            total_credit += abs(cost) * abs(qty) * 100
        else:
            total_credit -= cost * qty * 100

    # Breakeven points
    short_put_strike = put_spread.get('short_strike', 0)
    short_call_strike = call_spread.get('short_strike', 0)

    # Net credit per share (total_credit is for 1 contract = 100 shares)
    net_credit_per_share = total_credit / 100 if total_credit else 0

    result['lower_be'] = short_put_strike - net_credit_per_share
    result['upper_be'] = short_call_strike + net_credit_per_share
    result['max_profit'] = total_credit

    # Max loss = width of wider spread * 100 - max_profit
    put_width = (put_spread.get('short_strike', 0) - put_spread.get('long_strike', 0)) * 100
    call_width = (call_spread.get('long_strike', 0) - call_spread.get('short_strike', 0)) * 100
    max_spread_width = max(put_width, call_width)
    result['max_loss'] = max_spread_width - total_credit

    # Risk assessment
    if spx_price:
        if result['lower_be'] and result['upper_be']:
            if result['lower_be'] <= spx_price <= result['upper_be']:
                result['risk_level'] = 'safe (in profit zone)'
            elif spx_price < result['lower_be']:
                dist = result['lower_be'] - spx_price
                result['risk_level'] = f"warning (SPX {dist:.0f} pts below lower BE)"
            else:
                dist = spx_price - result['upper_be']
                result['risk_level'] = f"warning (SPX {dist:.0f} pts above upper BE)"

    return result


def generate_recommendations(strategies: dict, spx_price: float, vix_price: float) -> list:
    """Generate adjustment recommendations"""
    recommendations = []

    for expiry, strategy in strategies.items():
        analysis = analyze_iron_condor(strategy, spx_price)

        rec = {
            'expiry': expiry,
            'strategy': strategy.get('name', 'Unknown'),
            'actions': [],
        }

        if 'Iron Condor' in strategy.get('name', ''):
            if analysis['risk_level'] == 'safe (in profit zone)':
                rec['actions'].append("Hold - current price is in profit zone")
                rec['actions'].append("Monitor VIX, consider selling more premium if it drops")
            elif 'below' in analysis.get('risk_level', ''):
                rec['actions'].append("URGENT: Put side under test, consider:")
                rec['actions'].append("  a) Roll down put side to collect more premium")
                rec['actions'].append("  b) Close put side to limit loss, keep call side")
                rec['actions'].append("  c) Add long put hedge")
            elif 'above' in analysis.get('risk_level', ''):
                rec['actions'].append("URGENT: Call side under test, consider:")
                rec['actions'].append("  a) Roll up call side to collect more premium")
                rec['actions'].append("  b) Close call side to limit loss")

        recommendations.append(rec)

    # Overall market recommendation
    if vix_price and vix_price > 30:
        recommendations.append({
            'type': 'market',
            'message': f"VIX {vix_price:.1f} is HIGH - consider adding hedges or reducing risk exposure"
        })
    elif vix_price and vix_price < 20:
        recommendations.append({
            'type': 'market',
            'message': f"VIX {vix_price:.1f} is LOW - favorable for selling options premium"
        })

    return recommendations


# ============== Main Function ==============

def main():
    parser = argparse.ArgumentParser(
        description='Get portfolio advice - SPX/VIX market data + SPX options analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python get_portfolio_advice.py
    python get_portfolio_advice.py --trd-env SIMULATE
    python get_portfolio_advice.py --json
        """
    )
    parser.add_argument('--trd-env', default='REAL', choices=['REAL', 'SIMULATE'],
                        help='Trading environment')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    parser.add_argument('--no-positions', action='store_true',
                        help='Skip fetching positions (market data only)')

    args = parser.parse_args()

    # Step 1: Get market data
    print("Fetching market data...", file=sys.stderr)
    market_data = get_market_data(['SPX', 'VIX'])

    spx_info = market_data.get('SPX', {})
    vix_info = market_data.get('VIX', {})

    spx_price = spx_info.get('price')
    vix_price = vix_info.get('price')

    if not args.json:
        print("=" * 80)
        print("MARKET OVERVIEW")
        print("=" * 80)

        if 'error' not in spx_info and spx_price:
            print(f"\nSPX - {spx_info.get('name', 'S&P 500')}")
            print(f"  Price: ${spx_price:.2f}")
            print(f"  Change: {spx_info.get('change', 'N/A')} ({spx_info.get('change_percent', 'N/A')}%)")
            print(f"  Range: {spx_info.get('day_low', 'N/A')} - {spx_info.get('day_high', 'N/A')}")
        else:
            print(f"\nSPX: Failed to fetch - {spx_info.get('error', 'Unknown error')}")

        if 'error' not in vix_info and vix_price:
            print(f"\nVIX - {vix_info.get('name', 'CBOE Volatility Index')}")
            print(f"  Price: ${vix_price:.2f}")
            print(f"  Change: {vix_info.get('change', 'N/A')} ({vix_info.get('change_percent', 'N/A')}%)")

            # VIX interpretation
            if vix_price > 30:
                print("  Status: HIGH - Market fear elevated")
            elif vix_price > 25:
                print("  Status: ELEVATED - Some market concern")
            elif vix_price > 20:
                print("  Status: MODERATE - Normal market conditions")
            else:
                print("  Status: LOW - Calm market, good for selling premium")
        else:
            print(f"\nVIX: Failed to fetch - {vix_info.get('error', 'Unknown error')}")

    # Step 2: Get and analyze positions
    if args.no_positions:
        if args.json:
            output = {
                'market_data': market_data,
                'message': 'Positions not fetched (--no-positions flag)'
            }
            print(json.dumps(output, indent=2, default=str))
        return

    print("\nFetching portfolio positions...", file=sys.stderr)
    positions = get_portfolio_positions(args.trd_env)

    if positions is None or positions.empty:
        if args.json:
            output = {
                'market_data': market_data,
                'error': 'No positions found or failed to fetch'
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            print("\nNo positions found or failed to fetch from Futu OpenAPI")
            print("Make sure OpenD is running and connected to your account")
        return

    # Filter SPX options
    spx_options = positions[positions['code'].str.contains('SPX', case=False)]

    if spx_options.empty:
        if args.json:
            output = {
                'market_data': market_data,
                'message': 'No SPX options positions found'
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            print("\nNo SPX options positions found")
        return

    # Step 3: Identify strategies
    print("Analyzing strategies...", file=sys.stderr)
    strategies = identify_strategy(spx_options)

    # Step 4: Generate output
    if args.json:
        output = {
            'market_data': market_data,
            'strategies': strategies,
            'recommendations': generate_recommendations(strategies, spx_price, vix_price),
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print("\n" + "=" * 80)
        print("SPX OPTIONS HOLDINGS ANALYSIS")
        print("=" * 80)

        for expiry, strategy in strategies.items():
            print("-" * 80)
            print(f"Expiry: {expiry}")
            print(f"Strategy: {strategy.get('name', 'Unknown')}")
            print(f"Total P&L: ${strategy['total_pl']:.2f}")

            if 'is_iron_condor' in strategy:
                analysis = analyze_iron_condor(strategy, spx_price)
                print(f"Breakeven Range: {analysis['lower_be']:.0f} - {analysis['upper_be']:.0f}")
                print(f"Max Profit: ${analysis['max_profit']:.0f}")
                print(f"Risk Level: {analysis['risk_level']}")

            # Position details
            print("\nPositions:")
            for leg in strategy.get('puts', []):
                side = "Long" if leg['qty'] > 0 else "Short"
                print(f"  {side} {leg['strike']}P x{abs(leg['qty'])}: cost={leg['cost_price']:.2f}, market={leg['market_val']:.2f}")
            for leg in strategy.get('calls', []):
                side = "Long" if leg['qty'] > 0 else "Short"
                print(f"  {side} {leg['strike']}C x{abs(leg['qty'])}: cost={leg['cost_price']:.2f}, market={leg['market_val']:.2f}")
            print()

        # Recommendations
        print("=" * 80)
        print("INVESTMENT RECOMMENDATIONS")
        print("=" * 80)
        recs = generate_recommendations(strategies, spx_price, vix_price)
        for rec in recs:
            if 'type' in rec and rec['type'] == 'market':
                print(f"\n[MARKET] {rec['message']}")
            else:
                print(f"\n{rec['expiry']} - {rec['strategy']}:")
                for action in rec['actions']:
                    print(f"  • {action}")

        print("\n" + "=" * 80)
        print("End of Report")
        print("=" * 80)


if __name__ == '__main__':
    main()
