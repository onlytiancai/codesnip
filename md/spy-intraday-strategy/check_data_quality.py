#!/usr/bin/env python3
"""
SPY数据质量检查脚本
检查下载的数据完整性、格式正确性、异常值等
"""

import pandas as pd
import numpy as np
import os

def check_data_quality(csv_file):
    """
    检查CSV数据质量
    """
    print("=" * 60)
    print(f"数据质量检查: {csv_file}")
    print("=" * 60)

    if not os.path.exists(csv_file):
        print(f"文件不存在: {csv_file}")
        return False

    # 读取数据
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

    print(f"\n【基本信息】")
    print(f"  总行数: {len(df):,}")
    print(f"  列名: {df.columns.tolist()}")
    print(f"  数据类型:\n{df.dtypes.to_string()}")

    # 时间范围检查
    print(f"\n【时间范围】")
    print(f"  开始时间: {df.index[0]}")
    print(f"  结束时间: {df.index[-1]}")

    # 缺失值检查
    print(f"\n【缺失值检查】")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ 无缺失值")
    else:
        print(f"  ✗ 存在缺失值:\n{missing[missing > 0].to_string()}")

    # 重复时间检查
    print(f"\n【重复时间检查】")
    duplicates = df.index.duplicated().sum()
    if duplicates == 0:
        print("  ✓ 无重复时间戳")
    else:
        print(f"  ✗ 存在 {duplicates} 个重复时间戳")

    # 排序检查
    print(f"\n【排序检查】")
    is_sorted = df.index.is_monotonic_increasing
    if is_sorted:
        print("  ✓ 时间戳已按升序排列")
    else:
        print("  ✗ 时间戳未按升序排列")

    # 价格异常检查
    print(f"\n【价格异常检查】")
    if 'Close' in df.columns:
        close = df['Close']
        high = df['High'] if 'High' in df.columns else close
        low = df['Low'] if 'Low' in df.columns else close
        open_col = df['Open'] if 'Open' in df.columns else close

        # 检查价格为0或负数
        invalid_close = (close <= 0).sum()
        if invalid_close > 0:
            print(f"  ✗ Close存在 {invalid_close} 个异常值(<=0)")
        else:
            print(f"  ✓ Close无异常值(<=0)")

        # 检查High < Low
        if 'High' in df.columns and 'Low' in df.columns:
            invalid_hl = (high < low).sum()
            if invalid_hl > 0:
                print(f"  ✗ High < Low: {invalid_hl} 条")
            else:
                print(f"  ✓ High >= Low")

        # 检查Close超出High-Low范围
        invalid_range = ((close > high) | (close < low)).sum()
        if invalid_range > 0:
            print(f"  ✗ Close超出High-Low范围: {invalid_range} 条")
        else:
            print(f"  ✓ Close在High-Low范围内")

        # 检查Open超出High-Low范围
        if 'Open' in df.columns:
            invalid_open = ((open_col > high) | (open_col < low)).sum()
            if invalid_open > 0:
                print(f"  ✗ Open超出High-Low范围: {invalid_open} 条")
            else:
                print(f"  ✓ Open在High-Low范围内")

    # 成交量检查
    print(f"\n【成交量检查】")
    if 'Volume' in df.columns:
        vol = df['Volume']
        zero_vol = (vol == 0).sum()
        neg_vol = (vol < 0).sum()
        print(f"  零成交量: {zero_vol} 条")
        print(f"  负成交量: {neg_vol} 条")
        if zero_vol > len(df) * 0.1:
            print(f"  ⚠ 零成交量比例较高: {zero_vol/len(df)*100:.1f}%")
        else:
            print(f"  ✓ 成交量数据正常")

    # 价格跳变检查
    print(f"\n【价格跳变检查】")
    if 'Close' in df.columns:
        close = df['Close']
        pct_change = close.pct_change().abs()
        large_jumps = (pct_change > 0.1).sum()  # 10%以上跳变
        print(f"  10%以上跳变: {large_jumps} 条")
        if large_jumps > 0:
            print(f"  ⚠ 存在较大价格跳变，可能为数据问题")
            # 显示跳变详情
            jump_idx = pct_change[pct_change > 0.1].index
            for idx in jump_idx[:5]:  # 最多显示5个
                print(f"    {idx}: {close[idx]*100:.2f}%")

    # 统计摘要
    print(f"\n【价格统计摘要】")
    if 'Close' in df.columns:
        close = df['Close']
        print(f"  Close均值: {close.mean():.2f}")
        print(f"  Close标准差: {close.std():.2f}")
        print(f"  Close最小值: {close.min():.2f}")
        print(f"  Close最大值: {close.max():.2f}")

    if 'Volume' in df.columns:
        vol = df['Volume']
        print(f"  Volume均值: {vol.mean():,.0f}")
        print(f"  Volume标准差: {vol.std():,.0f}")

    # 时间间隔检查（对于分钟/小时数据）
    print(f"\n【时间间隔检查】")
    try:
        time_diffs = pd.to_datetime(df.index).to_series().diff().dropna()
        most_common_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else None
        irregular_intervals = (time_diffs != most_common_diff).sum() if most_common_diff else 0
        print(f"  常见间隔: {most_common_diff}")
        print(f"  不规则间隔数: {irregular_intervals}")
        if irregular_intervals > len(df) * 0.05:
            print(f"  ⚠ 不规则间隔比例较高: {irregular_intervals/len(df)*100:.1f}%")
    except Exception as e:
        print(f"  ⚠ 时间间隔检查失败: {e}")

    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)

    return True

def check_all_data(data_dir='.'):
    """
    检查目录下所有数据文件
    """
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f.startswith('spy')]

    if not csv_files:
        print(f"在 {data_dir} 中未找到SPY数据文件")
        return

    print(f"\n找到 {len(csv_files)} 个数据文件:\n")

    for csv_file in sorted(csv_files):
        file_path = os.path.join(data_dir, csv_file)
        check_data_quality(file_path)
        print("\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='检查SPY数据质量')
    parser.add_argument('file', nargs='?', help='CSV文件路径')
    parser.add_argument('--dir', default='.', help='检查目录下的所有文件')

    args = parser.parse_args()

    if args.file:
        check_data_quality(args.file)
    else:
        check_all_data(args.dir)