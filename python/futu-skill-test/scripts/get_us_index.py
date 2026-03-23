#!/usr/bin/env python3
"""
获取美股指数行情数据（通过 yfinance 替代方案）

当用户无美股行情权限时，可使用此脚本通过雅虎财经 API 获取 SPX、VIX、NDX 等指数数据。

用法:
    python get_us_index.py SPX VIX NDX [--period 1mo]

支持代码:
    - SPX: 标普 500 指数
    - VIX: 恐慌指数
    - NDX: 纳斯达克 100 指数
    - DJI: 道琼斯工业平均指数
"""

import sys
import argparse
import json

try:
    import yfinance as yf
except ImportError:
    print("错误：需要安装 yfinance，请运行：pip install yfinance")
    sys.exit(1)


INDEX_MAP = {
    'SPX': '^SPX',
    '^SPX': '^SPX',
    'VIX': '^VIX',
    '^VIX': '^VIX',
    'NDX': '^NDX',
    '^NDX': '^NDX',
    'DJI': '^DJI',
    '^DJI': '^DJI',
    'RUT': '^RUT',
    '^RUT': '^RUT',
}


def get_index_info(symbol: str) -> dict:
    """获取指数信息"""
    ticker = INDEX_MAP.get(symbol, symbol)
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


def get_history(symbol: str, period: str = '1mo') -> list:
    """获取历史走势"""
    ticker = INDEX_MAP.get(symbol, symbol)
    try:
        idx = yf.Ticker(ticker)
        hist = idx.history(period=period)
        result = []
        for date, row in hist.iterrows():
            result.append({
                'date': str(date),
                'close': row.get('Close'),
                'open': row.get('Open'),
                'high': row.get('High'),
                'low': row.get('Low'),
                'volume': row.get('Volume'),
            })
        return result
    except Exception as e:
        return [{'error': str(e)}]


def main():
    parser = argparse.ArgumentParser(description='获取美股指数行情（yfinance 替代方案）')
    parser.add_argument('symbols', nargs='*', default=['SPX', 'VIX'],
                        help='指数代码：SPX, VIX, NDX, DJI')
    parser.add_argument('--period', default='1mo',
                        help='历史数据周期：1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max')
    parser.add_argument('--history', action='store_true', help='获取历史走势')
    parser.add_argument('--json', action='store_true', help='JSON 格式输出')

    args = parser.parse_args()

    results = []

    for symbol in args.symbols:
        info = get_index_info(symbol)

        if args.history:
            info['history'] = get_history(symbol, args.period)

        results.append(info)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for r in results:
            print('=' * 60)
            if 'error' in r and len(r) == 1:
                print(f"{r['symbol']}: 获取失败 - {r.get('error', '未知错误')}")
            else:
                print(f"{r['symbol']} - {r.get('name', 'N/A')}")
                print(f"  当前价格：{r.get('price', 'N/A')}")
                print(f"  涨跌：{r.get('change', 'N/A')} ({r.get('change_percent', 'N/A')}%)")
                print(f"  开盘：{r.get('open', 'N/A')}")
                print(f"  昨收：{r.get('previous_close', 'N/A')}")
                print(f"  日内范围：{r.get('day_low', 'N/A')} - {r.get('day_high', 'N/A')}")
                print(f"  52 周范围：{r.get('52_week_low', 'N/A')} - {r.get('52_week_high', 'N/A')}")
                print(f"  成交量：{r.get('volume', 'N/A')}")

                if 'history' in r and r['history']:
                    print(f"\n  近 {args.period} 走势:")
                    for h in r['history'][-5:]:
                        if 'error' not in h:
                            print(f"    {h['date'][:10]}: {h['close']:.2f}")


if __name__ == '__main__':
    main()
