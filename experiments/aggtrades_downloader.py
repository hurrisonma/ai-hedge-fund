#!/usr/bin/env python3
"""
📊 币安aggTrades数据下载器
专门下载USDCUSDT逐笔交易数据并聚合为稳定币预测特征

数据源：https://data.binance.vision/data/spot/monthly/aggTrades/USDCUSDT/
"""

import os
import time
import zipfile
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


class AggTradesDownloader:
    """币安aggTrades数据下载和聚合器"""
    
    def __init__(self, symbol: str = "USDCUSDT"):
        self.symbol = symbol
        self.base_url = "https://data.binance.vision/data/spot/monthly/aggTrades"
        self.raw_dir = "data/raw/aggtrades"
        self.processed_dir = "data/processed"
        
        # 创建目录
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        print("📊 币安aggTrades下载器初始化")
        print(f"交易对: {self.symbol}")
        print(f"数据源: {self.base_url}")
        print(f"原始数据目录: {self.raw_dir}")
        print(f"处理后目录: {self.processed_dir}")
    
    def download_month_data(self, year: int, month: int) -> str:
        """下载单个月份的aggTrades数据"""
        print(f"\n📅 下载 {year}-{month:02d} 月aggTrades数据")
        print("-" * 50)
        
        # 生成文件名和URL
        filename = f"{self.symbol}-aggTrades-{year}-{month:02d}.zip"
        url = f"{self.base_url}/{self.symbol}/{filename}"
        zip_path = os.path.join(self.raw_dir, filename)
        
        # 检查是否已存在
        if os.path.exists(zip_path):
            print(f"✅ 文件已存在: {filename}")
            return zip_path
        
        try:
            print(f"📥 开始下载: {filename}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ 下载完成: {zip_path}")
            return zip_path
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return ""
    
    def extract_and_process(self, zip_path: str) -> pd.DataFrame:
        """解压并处理aggTrades数据"""
        if not zip_path or not os.path.exists(zip_path):
            return pd.DataFrame()
        
        try:
            print(f"📂 解压和处理: {os.path.basename(zip_path)}")
            
            # 解压文件
            extract_dir = os.path.dirname(zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                if len(file_list) != 1:
                    print(f"❌ ZIP文件包含多个文件: {file_list}")
                    return pd.DataFrame()
                
                csv_filename = file_list[0]
                zip_ref.extractall(extract_dir)
                csv_path = os.path.join(extract_dir, csv_filename)
            
            # 读取CSV数据
            print(f"📖 读取CSV文件: {csv_filename}")
            
            # aggTrades数据列名
            columns = [
                'agg_trade_id', 'price', 'quantity', 'first_trade_id', 
                'last_trade_id', 'timestamp', 'is_buyer_maker', 
                'is_best_match'
            ]
            
            df = pd.read_csv(csv_path, names=columns)
            print(f"📊 原始数据行数: {len(df)}")
            
            # 数据类型转换
            df['price'] = pd.to_numeric(df['price'])
            df['quantity'] = pd.to_numeric(df['quantity'])
            
            # 修复时间戳格式 - aggTrades使用微秒时间戳
            # 检测时间戳位数并正确转换
            sample_timestamp = df['timestamp'].iloc[0]
            if len(str(sample_timestamp)) >= 16:
                # 16位微秒时间戳，需要除以1000转为毫秒
                df['timestamp'] = pd.to_datetime(df['timestamp'] / 1000, unit='ms')
            else:
                # 13位毫秒时间戳
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df['is_buyer_maker'] = df['is_buyer_maker'].astype(bool)
            
            time_min = df.timestamp.min()
            time_max = df.timestamp.max()
            print(f"🕒 时间范围: {time_min} 到 {time_max}")
            
            # 删除解压的CSV文件
            os.remove(csv_path)
            
            return df
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            return pd.DataFrame()
    
    def aggregate_to_minute(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """将逐笔交易聚合为分钟级特征"""
        if trades_df.empty:
            return pd.DataFrame()
        
        print(f"🔄 聚合 {len(trades_df)} 条交易记录到分钟级...")
        
        # 设置时间索引
        trades_df = trades_df.set_index('timestamp')
        
        # 按分钟分组聚合
        result_list = []
        
        for minute_start, group in trades_df.groupby(pd.Grouper(freq='1min')):
            if len(group) == 0:
                continue
            
            # 分离买卖交易
            buy_trades = group[~group['is_buyer_maker']]  # 主动买入
            sell_trades = group[group['is_buyer_maker']]  # 主动卖出
            
            # 价格特征
            prices = group['price']
            best_bid = (buy_trades['price'].max() if len(buy_trades) > 0
                       else prices.max())
            best_ask = (sell_trades['price'].min() if len(sell_trades) > 0
                       else prices.min())
            spread = best_ask - best_bid
            
            # 成交量特征
            total_volume = group['quantity'].sum()
            buy_volume = (buy_trades['quantity'].sum() if len(buy_trades) > 0
                         else 0)
            sell_volume = (sell_trades['quantity'].sum() 
                          if len(sell_trades) > 0 else 0)
            buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
            
            # 流动性特征
            trade_count = len(group)
            avg_trade_size = (total_volume / trade_count 
                             if trade_count > 0 else 0)
            
            # 大单检测（前20%视为大单）
            large_threshold = (group['quantity'].quantile(0.8) 
                              if len(group) > 5 
                              else group['quantity'].median())
            large_trades = group[group['quantity'] > large_threshold]
            large_trade_count = len(large_trades)
            large_trade_ratio = (large_trade_count / trade_count 
                                if trade_count > 0 else 0)
            
            # 价格冲击
            price_range = prices.max() - prices.min()
            volume_impact = (price_range / total_volume 
                            if total_volume > 0 else 0)
            
            result_list.append({
                'timestamp': minute_start,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'total_volume': total_volume,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_ratio': buy_ratio,
                'trade_count': trade_count,
                'avg_trade_size': avg_trade_size,
                'large_trade_count': large_trade_count,
                'large_trade_ratio': large_trade_ratio,
                'price_range': price_range,
                'volume_impact': volume_impact
            })
        
        result_df = pd.DataFrame(result_list)
        print(f"✅ 聚合完成: {len(result_df)} 分钟记录")
        
        return result_df
    
    def download_and_aggregate_month(self, year: int, month: int) -> pd.DataFrame:
        """下载并聚合单个月份数据"""
        # 下载原始数据
        zip_path = self.download_month_data(year, month)
        if not zip_path:
            return pd.DataFrame()
        
        # 解压处理
        trades_df = self.extract_and_process(zip_path)
        if trades_df.empty:
            return pd.DataFrame()
        
        # 聚合为分钟级
        minute_df = self.aggregate_to_minute(trades_df)
        
        # 保存月份聚合数据
        if not minute_df.empty:
            month_file = f"{self.symbol}_aggTrades_{year}_{month:02d}.csv"
            month_path = os.path.join(self.processed_dir, month_file)
            minute_df.to_csv(month_path, index=False)
            
            file_size = os.path.getsize(month_path) / 1024 / 1024
            print(f"💾 月份数据已保存: {month_path} ({file_size:.2f}MB)")
        
        return minute_df
    
    def download_recent_months(self, months: int = 6) -> str:
        """下载最近几个月的aggTrades数据"""
        print(f"🚀 开始下载最近 {months} 个月的aggTrades数据")
        print("=" * 60)
        
        # 计算月份范围
        end_date = datetime.now() - timedelta(days=30)
        start_date = end_date - timedelta(days=months * 30)
        
        # 生成月份列表
        current_date = datetime(start_date.year, start_date.month, 1)
        end_check = datetime(end_date.year, end_date.month, 1)
        
        months_to_download = []
        while current_date <= end_check:
            months_to_download.append((current_date.year, current_date.month))
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        print(f"📋 需要下载的月份: {[f'{y}-{m:02d}' for y, m in months_to_download]}")
        
        # 下载和聚合各月数据
        all_month_data = []
        success_count = 0
        
        for year, month in months_to_download:
            month_df = self.download_and_aggregate_month(year, month)
            if not month_df.empty:
                all_month_data.append(month_df)
                success_count += 1
                print(f"✅ {year}-{month:02d} 处理成功")
            else:
                print(f"❌ {year}-{month:02d} 处理失败")
            
            # 添加延迟避免请求过快
            time.sleep(1)
        
        print(f"\n📊 下载汇总: {success_count}/{len(months_to_download)} 个月份成功")
        
        if not all_month_data:
            print("❌ 没有成功下载任何数据")
            return ""
        
        # 合并所有月份数据
        print("🔗 合并所有月份数据...")
        combined_df = pd.concat(all_month_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # 保存最终数据
        output_filename = f"{self.symbol}_aggTrades_recent_{months}months.csv"
        output_path = os.path.join(self.processed_dir, output_filename)
        combined_df.to_csv(output_path, index=False)
        
        # 统计信息
        file_size = os.path.getsize(output_path) / 1024 / 1024
        time_span = (combined_df['timestamp'].max() - combined_df['timestamp'].min()).days
        
        print(f"\n✅ 最终数据保存完成!")
        print(f"📄 输出文件: {output_path}")
        print(f"📏 文件大小: {file_size:.2f} MB")
        print(f"📊 总分钟数: {len(combined_df):,}")
        print(f"📅 时间跨度: {time_span} 天")
        print(f"🕒 时间范围: {combined_df['timestamp'].min()} 到 {combined_df['timestamp'].max()}")
        
        return output_path


def main():
    """主函数"""
    print("📊 币安aggTrades数据下载器")
    print("=" * 50)
    
    downloader = AggTradesDownloader("USDCUSDT")
    
    print("\n请选择下载方式:")
    print("1. 下载最近6个月aggTrades数据（推荐）")
    print("2. 下载最近3个月aggTrades数据")
    print("3. 下载单个月份数据")
    
    try:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            # 最近6个月
            output_path = downloader.download_recent_months(6)
            if output_path:
                print(f"\n🎉 下载完成！数据文件: {output_path}")
        
        elif choice == "2":
            # 最近3个月
            output_path = downloader.download_recent_months(3)
            if output_path:
                print(f"\n🎉 下载完成！数据文件: {output_path}")
        
        elif choice == "3":
            # 单个月份
            year = int(input("请输入年份 (如 2025): "))
            month = int(input("请输入月份 (1-12): "))
            
            month_df = downloader.download_and_aggregate_month(year, month)
            if not month_df.empty:
                print(f"\n🎉 {year}-{month:02d} 月数据下载完成！")
        
        else:
            print("❌ 无效选择")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  用户取消下载")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")


if __name__ == "__main__":
    main() 