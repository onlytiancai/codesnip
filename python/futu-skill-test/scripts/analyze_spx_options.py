#!/usr/bin/env python3
"""
分析 SPX 期权持仓策略

自动识别铁鹰策略 (Iron Condor)、垂直价差 (Vertical Spread) 等期权组合，
计算盈亏平衡点、最大盈利/亏损、当前盈亏，并给出调整建议。

用法:
    python analyze_spx_options.py [--json]
"""

import sys
import json
import argparse
from datetime import datetime

try:
    from futu import *
except ImportError:
    print("错误：需要安装 futu-api，请运行：pip install futu-api")
    sys.exit(1)


def get_portfolio_positions(trd_env: str = 'REAL'):
    """获取持仓数据"""
    try:
        trd_ctx = OpenSecTradeContext(host='127.0.0.1', port=11111)
        trd_env_enum = TrdEnv.REAL if trd_env == 'REAL' else TrdEnv.SIMULATE
        ret, data = trd_ctx.position_list_query(trd_env=trd_env_enum)
        trd_ctx.close()

        if ret == RET_OK:
            return data
        else:
            print(f"获取持仓失败：{data}")
            return None
    except Exception as e:
        print(f"错误：{e}")
        return None


def identify_strategy(positions: list) -> dict:
    """
    识别期权策略类型

    支持的策略:
    - Iron Condor (铁鹰): Short OTM Put + Long OTM Put + Short OTM Call + Long OTM Call
    - Put Vertical Spread: Long Put + Short Put (同到期日)
    - Call Vertical Spread: Short Call + Long Call (同到期日)
    - Straddle: Long/Short Call + Long/Short Put (同行权价)
    - Strangle: Long/Short Call + Long/Short Put (不同行权价)
    """
    strategies = {}

    # 按到期日分组
    by_expiry = {}
    for _, pos in positions.iterrows():
        code = pos['code']
        # 解析期权代码格式：US.SPX260417P6500000
        if code.startswith('US.SPX') or code.startswith('US..SPX'):
            # 提取到期日和类型
            match = None
            if 'P' in code.split('SPX')[1]:
                parts = code.split('SPX')
                if len(parts) > 1:
                    rest = parts[1]
                    # 提取日期
                    date_str = rest[:6]
                    type_char = rest[6] if len(rest) > 6 else None
                    strike = float(rest[7:]) / 1000 if len(rest) > 7 else None

                    if date_str and type_char and strike:
                        expiry = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
                        key = f"{expiry}_{type_char}_{strike}"
                        if expiry not in by_expiry:
                            by_expiry[expiry] = {'puts': [], 'calls': []}

                        if type_char == 'P':
                            by_expiry[expiry]['puts'].append({
                                'code': code,
                                'qty': pos['qty'],
                                'cost_price': pos['cost_price'],
                                'market_val': pos['market_val'],
                                'pl_val': pos.get('pl_val', 0),
                                'strike': strike,
                            })
                        elif type_char == 'C':
                            by_expiry[expiry]['calls'].append({
                                'code': code,
                                'qty': pos['qty'],
                                'cost_price': pos['cost_price'],
                                'market_val': pos['market_val'],
                                'pl_val': pos.get('pl_val', 0),
                                'strike': strike,
                            })
            elif 'C' in code.split('SPX')[1]:
                parts = code.split('SPX')
                if len(parts) > 1:
                    rest = parts[1]
                    date_str = rest[:6]
                    type_char = rest[6] if len(rest) > 6 else None
                    strike = float(rest[7:]) / 1000 if len(rest) > 7 else None

                    if date_str and type_char and strike:
                        expiry = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
                        if expiry not in by_expiry:
                            by_expiry[expiry] = {'puts': [], 'calls': []}

                        if type_char == 'C':
                            by_expiry[expiry]['calls'].append({
                                'code': code,
                                'qty': pos['qty'],
                                'cost_price': pos['cost_price'],
                                'market_val': pos['market_val'],
                                'pl_val': pos.get('pl_val', 0),
                                'strike': strike,
                            })

    # 分析每个到期日的策略
    for expiry, legs in by_expiry.items():
        strategy_name = []
        details = {
            'expiry': expiry,
            'puts': legs['puts'],
            'calls': legs['calls'],
            'total_pl': 0,
        }

        # 检查 Put 端
        if legs['puts']:
            long_puts = [p for p in legs['puts'] if p['qty'] > 0]
            short_puts = [p for p in legs['puts'] if p['qty'] < 0]

            if long_puts and short_puts:
                # Put Vertical Spread
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

        # 检查 Call 端
        if legs['calls']:
            long_calls = [c for c in legs['calls'] if c['qty'] > 0]
            short_calls = [c for c in legs['calls'] if c['qty'] < 0]

            if long_calls and short_calls:
                # Call Vertical Spread
                long_strike = long_calls[0]['strike']
                short_strike = short_calls[0]['strike']
                if short_strike > long_strike:
                    strategy_name.append(f"Bear Call Spread {short_strike}/{long_strike}")
                else:
                    strategy_name.append(f"Bull Call Spread {long_strike}/{short_strike}")
                details['call_spread'] = {
                    'long_strike': long_strike if long_calls else None,
                    'short_strike': short_strike if short_calls else None,
                    'long_qty': sum(c['qty'] for c in long_calls),
                    'short_qty': sum(c['qty'] for c in short_calls),
                }

        # 检查是否构成铁鹰
        if legs['puts'] and legs['calls']:
            long_puts = [p for p in legs['puts'] if p['qty'] > 0]
            short_puts = [p for p in legs['puts'] if p['qty'] < 0]
            long_calls = [c for c in legs['calls'] if c['qty'] > 0]
            short_calls = [c for c in legs['calls'] if c['qty'] < 0]

            if all([long_puts, short_puts, long_calls, short_calls]):
                strategy_name.insert(0, "Iron Condor")
                details['is_iron_condor'] = True

        # 计算总盈亏 - 使用 futu 返回的 pl_val 字段（更准确）
        for leg in legs['puts'] + legs['calls']:
            details['total_pl'] += leg.get('pl_val', leg['market_val'] - (leg['cost_price'] * abs(leg['qty']) * 100))

        if strategy_name:
            details['name'] = ' + '.join(strategy_name)
            strategies[expiry] = details

    return strategies


def get_spx_price() -> float:
    """获取 SPX 当前价格（通过 yfinance）"""
    try:
        import yfinance as yf
        spx = yf.Ticker('^SPX')
        info = spx.info
        return info.get('regularMarketPrice', None)
    except:
        return None


def analyze_iron_condor(strategy: dict, spx_price: float) -> dict:
    """分析铁鹰策略的盈亏平衡点和风险"""
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

    # 计算净权利金 - 从实际持仓成本计算
    # Short 仓位 (qty < 0): 收取权利金
    # Long 仓位 (qty > 0): 支付权利金
    total_credit = 0
    for leg in strategy.get('puts', []) + strategy.get('calls', []):
        qty = leg.get('qty', 0)
        cost = leg.get('cost_price', 0)
        if qty < 0:
            total_credit += abs(cost) * abs(qty) * 100
        else:
            total_credit -= cost * qty * 100

    # 盈亏平衡点
    short_put_strike = put_spread.get('short_strike', 0)
    short_call_strike = call_spread.get('short_strike', 0)

    # 每股净权利金
    net_credit_per_share = total_credit / 100 if total_credit else 0

    result['lower_be'] = short_put_strike - net_credit_per_share
    result['upper_be'] = short_call_strike + net_credit_per_share
    result['max_profit'] = total_credit

    # 最大亏损 = 较宽价差宽度 - 净权利金
    put_width = (put_spread.get('short_strike', 0) - put_spread.get('long_strike', 0)) * 100
    call_width = (call_spread.get('long_strike', 0) - call_spread.get('short_strike', 0)) * 100
    max_spread_width = max(put_width, call_width)
    result['max_loss'] = max_spread_width - total_credit

    # 风险评估
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
    """生成调整建议"""
    recommendations = []

    for expiry, strategy in strategies.items():
        analysis = analyze_iron_condor(strategy, spx_price)

        rec = {
            'expiry': expiry,
            'strategy': strategy.get('name', 'Unknown'),
            'actions': [],
        }

        # 根据风险等级给出建议
        if 'Iron Condor' in strategy.get('name', ''):
            if analysis['risk_level'] == 'safe (in profit zone)':
                rec['actions'].append("保持持有 - 当前价格在盈利区间内")
                rec['actions'].append("监控 VIX，若回落可考虑收取更多权利金")
            elif 'below' in analysis.get('risk_level', ''):
                rec['actions'].append("紧急：Put 端受测试，考虑以下操作:")
                rec['actions'].append("  a) 向下滚动 Put 端，收取额外权利金")
                rec['actions'].append("  b) 平仓 Put 端止损，保留 Call 端收益")
                rec['actions'].append("  c) 加仓 Long Put 作为对冲")
            elif 'above' in analysis.get('risk_level', ''):
                rec['actions'].append("紧急：Call 端受测试，考虑以下操作:")
                rec['actions'].append("  a) 向上滚动 Call 端，收取额外权利金")
                rec['actions'].append("  b) 平仓 Call 端止损")

        recommendations.append(rec)

    # 整体建议
    if vix_price and vix_price > 30:
        recommendations.append({
            'type': 'market',
            'message': f"VIX {vix_price:.1f} 处于高位，考虑增加对冲或减少风险敞口"
        })
    elif vix_price and vix_price < 20:
        recommendations.append({
            'type': 'market',
            'message': f"VIX {vix_price:.1f} 处于低位，有利于卖出期权收取权利金"
        })

    return recommendations


def main():
    parser = argparse.ArgumentParser(description='分析 SPX 期权持仓策略')
    parser.add_argument('--trd-env', default='REAL', choices=['REAL', 'SIMULATE'],
                        help='交易环境')
    parser.add_argument('--json', action='store_true', help='JSON 格式输出')

    args = parser.parse_args()

    # 获取持仓
    positions = get_portfolio_positions(args.trd_env)
    if positions is None or positions.empty:
        print("未找到持仓或获取失败")
        return

    # 过滤期权持仓
    options = positions[positions['code'].str.contains('SPX', case=False)]
    if options.empty:
        print("未找到 SPX 期权持仓")
        return

    # 识别策略
    strategies = identify_strategy(options)

    # 获取 SPX 和 VIX 价格
    spx_price = get_spx_price()
    try:
        import yfinance as yf
        vix = yf.Ticker('^VIX')
        vix_price = vix.info.get('regularMarketPrice', None)
    except:
        vix_price = None

    if not args.json:
        print("=" * 80)
        print("SPX 期权持仓策略分析")
        print("=" * 80)
        print(f"SPX: {spx_price:.2f}" if spx_price else "SPX: N/A")
        print(f"VIX: {vix_price:.2f}" if vix_price else "VIX: N/A")
        print()

        for expiry, strategy in strategies.items():
            print("-" * 80)
            print(f"到期日：{expiry}")
            print(f"策略：{strategy.get('name', 'Unknown')}")
            print(f"总盈亏：${strategy['total_pl']:.2f}")

            if 'is_iron_condor' in strategy:
                analysis = analyze_iron_condor(strategy, spx_price)
                print(f"盈亏平衡区间：{analysis['lower_be']:.0f} - {analysis['upper_be']:.0f}")
                print(f"最大盈利：${analysis['max_profit']:.0f}")
                print(f"风险等级：{analysis['risk_level']}")

            # 持仓明细
            print("\n持仓明细:")
            for leg in strategy.get('puts', []):
                side = "Long" if leg['qty'] > 0 else "Short"
                print(f"  {side} {leg['strike']}P x{abs(leg['qty'])}: 成本={leg['cost_price']:.2f}, 市值={leg['market_val']:.2f}")
            for leg in strategy.get('calls', []):
                side = "Long" if leg['qty'] > 0 else "Short"
                print(f"  {side} {leg['strike']}C x{abs(leg['qty'])}: 成本={leg['cost_price']:.2f}, 市值={leg['market_val']:.2f}")
            print()

        # 建议
        print("=" * 80)
        print("调整建议")
        print("=" * 80)
        recs = generate_recommendations(strategies, spx_price, vix_price)
        for rec in recs:
            if 'type' in rec and rec['type'] == 'market':
                print(f"\n市场：{rec['message']}")
            else:
                print(f"\n{rec['expiry']} - {rec['strategy']}:")
                for action in rec['actions']:
                    print(f"  • {action}")
    else:
        output = {
            'spx_price': spx_price,
            'vix_price': vix_price,
            'strategies': strategies,
            'recommendations': generate_recommendations(strategies, spx_price, vix_price),
        }
        print(json.dumps(output, indent=2, default=str))


if __name__ == '__main__':
    main()
