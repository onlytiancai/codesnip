#!/usr/bin/env python3
"""
SPY分钟级数据下载脚本
下载SPY历史分钟数据并保存为CSV格式

注意：Yahoo Finance限制
- 1分钟数据：最多只能获取最近7天
- 1小时数据：可获取最近730天
- 日线数据：可获取最近几年

策略：分段下载1小时数据，拼接后作为日内数据使用
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np

# 代理设置
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10808'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10808'

def download_spy_hourly_to_csv(output_file='spy_hourly.csv', days=365):
    """
    下载SPY小时数据并保存为CSV

    Yahoo 1小时数据可以获取最近730天
    """
    print(f"正在下载SPY最近{days}天小时数据...")

    ticker = yf.Ticker("SPY")

    # 使用history获取小时数据
    df = yf.download("SPY", period=f"{days}d", interval="1h", auto_adjust=True, progress=False)

    if df.empty:
        print("下载失败，数据为空")
        return None

    # 处理MultiIndex列名
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # 去重和排序
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()

    # 转换时区为美国时间（NYSE时间）
    df.index = df.index.tz_convert('America/New_York')

    # 保存CSV
    df.to_csv(output_file)
    print(f"数据已保存到 {output_file}")
    print(f"数据范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"总条数: {len(df)}")

    return df

def download_spy_1min_recent(output_file='spy_1min_recent.csv'):
    """
    下载SPY最近7天的1分钟数据
    Yahoo限制：每次最多7天
    """
    print("正在下载SPY最近7天1分钟数据...")

    all_data = []

    # 分天下载确保数据完整性
    for days_ago in range(7):
        period = str(days_ago + 1) + 'd'
        try:
            spy = yf.download('SPY', period=period, interval='1m', progress=False, auto_adjust=True)
            if spy is not None and not spy.empty:
                spy = spy[~spy.index.duplicated(keep='last')]
                all_data.append(spy)
                print(f"  第{7-days_ago}天: {len(spy)} 条")
            time.sleep(0.5)  # 避免请求过快
        except Exception as e:
            print(f"  第{7-days_ago}天下载失败: {e}")

    if not all_data:
        print("未能获取任何1分钟数据")
        return None

    # 合并数据
    combined = pd.concat(all_data)
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()

    # 处理MultiIndex列名
    if isinstance(combined.columns, pd.MultiIndex):
        combined.columns = [col[0] for col in combined.columns]

    # 转换时区
    combined.index = combined.index.tz_convert('America/New_York')

    # 保存CSV
    combined.to_csv(output_file)
    print(f"1分钟数据已保存到 {output_file}")
    print(f"数据范围: {combined.index[0]} 至 {combined.index[-1]}")
    print(f"总条数: {len(combined)}")

    return combined

def download_spy_daily_to_csv(output_file='spy_daily.csv', years=5):
    """
    下载SPY日线数据（用于计算指标和长期分析）
    """
    print(f"正在下载SPY最近{years}年日线数据...")

    spy = yf.Ticker("SPY")
    df = spy.history(period=f"{years}y", auto_adjust=True)

    if df.empty:
        print("下载失败")
        return None

    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()

    # 转换时区
    df.index = df.index.tz_convert('America/New_York')

    df.to_csv(output_file)
    print(f"日线数据已保存到 {output_file}")
    print(f"数据范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"总条数: {len(df)}")

    return df

def main():
    import argparse
    parser = argparse.ArgumentParser(description='下载SPY历史数据')
    parser.add_argument('--type', choices=['hourly', '1min', 'daily', 'all'],
                        default='all', help='数据类型')
    parser.add_argument('--days', type=int, default=365, help='小时数据天数')
    parser.add_argument('--years', type=int, default=5, help='日线数据年数')
    parser.add_argument('--output-dir', default='.', help='输出目录')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.type == 'hourly' or args.type == 'all':
        output = os.path.join(args.output_dir, 'spy_hourly.csv')
        download_spy_hourly_to_csv(output, args.days)
        time.sleep(2)

    if args.type == '1min' or args.type == 'all':
        output = os.path.join(args.output_dir, 'spy_1min_recent.csv')
        download_spy_1min_recent(output)
        time.sleep(2)

    if args.type == 'daily' or args.type == 'all':
        output = os.path.join(args.output_dir, 'spy_daily.csv')
        download_spy_daily_to_csv(output, args.years)

    print("\n下载完成!")

if __name__ == "__main__":
    main()