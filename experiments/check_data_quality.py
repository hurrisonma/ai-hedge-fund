#!/usr/bin/env python3
"""
📊 数据质量检查工具
检查下载的币安数据的时间连续性、价格合理性等质量指标
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def check_data_quality(file_path: str):
    """检查数据质量"""
    print("📊 加载数据...")
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"总记录数: {len(df):,}")
    print(f"时间范围: {df.timestamp.min()} 到 {df.timestamp.max()}")
    print(f"实际天数: {(df.timestamp.max() - df.timestamp.min()).days} 天")
    
    # 检查时间连续性
    print("\n🕒 时间连续性检查:")
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    time_diffs = df_sorted['timestamp'].diff().dropna()
    
    # 标准间隔应该是1分钟
    expected_interval = timedelta(minutes=1)
    normal_intervals = (time_diffs == expected_interval).sum()
    total_intervals = len(time_diffs)
    
    print(f"标准1分钟间隔: {normal_intervals:,}/{total_intervals:,} ({normal_intervals/total_intervals*100:.2f}%)")
    
    # 检查异常间隔
    abnormal_intervals = time_diffs[time_diffs != expected_interval]
    if len(abnormal_intervals) > 0:
        print(f"异常间隔数量: {len(abnormal_intervals)}")
        print(f"异常间隔统计:")
        
        # 按异常类型分类
        short_gaps = abnormal_intervals[abnormal_intervals < expected_interval]
        long_gaps = abnormal_intervals[abnormal_intervals > expected_interval]
        
        if len(short_gaps) > 0:
            print(f"  小于1分钟的间隔: {len(short_gaps)} 个")
        
        if len(long_gaps) > 0:
            print(f"  大于1分钟的间隔: {len(long_gaps)} 个")
            print(f"  最大间隔: {long_gaps.max()}")
            
            # 找出大间隔的具体位置
            large_gap_indices = time_diffs[time_diffs > timedelta(minutes=5)].index
            if len(large_gap_indices) > 0:
                print(f"  大于5分钟的间隔位置:")
                for idx in large_gap_indices[:5]:  # 显示前5个
                    before_time = df_sorted.iloc[idx-1]['timestamp']
                    after_time = df_sorted.iloc[idx]['timestamp']
                    gap = after_time - before_time
                    print(f"    {before_time} -> {after_time} (间隔: {gap})")
    
    # 检查重复时间戳
    duplicates = df['timestamp'].duplicated().sum()
    print(f"重复时间戳: {duplicates} 个")
    
    # 检查预期的总记录数
    expected_total = (df.timestamp.max() - df.timestamp.min()).total_seconds() / 60 + 1
    coverage = len(df) / expected_total * 100
    print(f"时间覆盖率: {coverage:.2f}% ({len(df):,}/{expected_total:.0f})")
    
    print("\n📈 价格数据质量检查:")
    print(f"开盘价范围: {df.open.min():.6f} - {df.open.max():.6f}")
    print(f"收盘价范围: {df.close.min():.6f} - {df.close.max():.6f}")
    print(f"最高价范围: {df.high.min():.6f} - {df.high.max():.6f}")
    print(f"最低价范围: {df.low.min():.6f} - {df.low.max():.6f}")
    
    # 价格变化幅度
    price_range = df.close.max() - df.close.min()
    avg_price = df.close.mean()
    price_volatility = price_range / avg_price * 100
    print(f"总价格变化幅度: {price_volatility:.4f}%")
    
    # 检查OHLC逻辑
    ohlc_errors = (
        (df.high < df.low) | 
        (df.high < df.open) | 
        (df.high < df.close) | 
        (df.low > df.open) | 
        (df.low > df.close)
    ).sum()
    print(f"OHLC逻辑错误: {ohlc_errors} 条")
    
    # 检查价格跳跃
    price_changes = df['close'].pct_change().abs()
    large_changes = price_changes[price_changes > 0.001]  # 0.1%以上变化
    print(f"大于0.1%的价格跳跃: {len(large_changes)} 次")
    if len(large_changes) > 0:
        print(f"最大单次价格变化: {price_changes.max()*100:.4f}%")
    
    print("\n📊 成交量统计:")
    volume_stats = df.volume.describe()
    print(f"平均成交量: {volume_stats['mean']:.0f}")
    print(f"中位数成交量: {volume_stats['50%']:.0f}")
    print(f"最大成交量: {volume_stats['max']:.0f}")
    print(f"最小成交量: {volume_stats['min']:.0f}")
    
    zero_volume = (df.volume == 0).sum()
    print(f"零成交量记录: {zero_volume} 条")
    
    # 异常成交量检测
    volume_q99 = df.volume.quantile(0.99)
    high_volume = (df.volume > volume_q99).sum()
    print(f"异常高成交量(>99%分位): {high_volume} 条")
    
    print("\n📅 按月份统计:")
    df['month'] = df['timestamp'].dt.to_period('M')
    monthly_stats = df.groupby('month').agg({
        'timestamp': 'count',
        'close': ['min', 'max', 'mean'],
        'volume': 'mean'
    }).round(6)
    
    monthly_stats.columns = ['记录数', '最低价', '最高价', '平均价', '平均成交量']
    print(monthly_stats)
    
    print("\n✅ 数据质量检查完成!")
    
    # 生成质量评分
    quality_score = 0
    max_score = 100
    
    # 时间连续性评分 (40分)
    continuity_score = min(40, (normal_intervals / total_intervals) * 40)
    quality_score += continuity_score
    
    # OHLC正确性评分 (20分)
    ohlc_score = 20 if ohlc_errors == 0 else max(0, 20 - ohlc_errors)
    quality_score += ohlc_score
    
    # 成交量合理性评分 (20分)
    volume_score = 20 if zero_volume == 0 else max(0, 20 - zero_volume)
    quality_score += volume_score
    
    # 价格合理性评分 (20分)
    price_score = 20 if price_volatility < 1.0 else max(0, 20 - price_volatility)
    quality_score += price_score
    
    print(f"\n🎯 数据质量评分: {quality_score:.1f}/{max_score}")
    
    if quality_score >= 90:
        print("🌟 数据质量: 优秀")
    elif quality_score >= 80:
        print("✅ 数据质量: 良好") 
    elif quality_score >= 70:
        print("⚠️ 数据质量: 一般")
    else:
        print("❌ 数据质量: 较差")
    
    return df

if __name__ == "__main__":
    file_path = "data/processed/USDCUSDT_recent_6months.csv"
    df = check_data_quality(file_path) 