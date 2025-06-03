#!/usr/bin/env python3
"""
📊 币安数据下载器
从币安官方数据网站下载真实的K线数据用于深度学习训练

支持功能：
- 自动下载指定时间范围的1分钟K线数据
- 数据清洗和格式转换
- 合并多个月份数据
- 数据质量检查
"""

import os
import time
import zipfile
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm


class BinanceDataDownloader:
    """币安数据下载器"""
    
    def __init__(self, symbol: str = "USDCUSDT", interval: str = "1m"):
        self.symbol = symbol
        self.interval = interval
        self.base_url = "https://data.binance.vision/data/spot/monthly/klines"
        self.data_dir = "data/raw/binance"
        self.processed_dir = "data/processed"
        
        # 创建目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        print("📊 币安数据下载器初始化")
        print(f"交易对: {self.symbol}")
        print(f"时间间隔: {self.interval}")
        print(f"数据目录: {self.data_dir}")
    
    def generate_download_urls(self, start_year: int, start_month: int, 
                               end_year: int, end_month: int) -> List[str]:
        """生成下载链接列表，避免未来日期"""
        urls = []
        
        current_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 1)
        
        # 确保不会下载未来的数据
        now = datetime.now()
        max_date = datetime(now.year, now.month, 1)
        if end_date > max_date:
            end_date = max_date
            print(f"⚠️  调整结束日期到当前月份: {max_date.strftime('%Y-%m')}")
        
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            
            # 构建文件名和URL
            filename = f"{self.symbol}-{self.interval}-{year}-{month:02d}.zip"
            url = f"{self.base_url}/{self.symbol}/{self.interval}/{filename}"
            urls.append(url)
            
            # 下个月
            if month == 12:
                current_date = datetime(year + 1, 1, 1)
            else:
                current_date = datetime(year, month + 1, 1)
        
        return urls
    
    def download_file(self, url: str, save_path: str) -> bool:
        """下载单个文件"""
        try:
            print(f"📥 下载: {os.path.basename(url)}")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            
            with open(save_path, 'wb') as f, tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ 下载完成: {save_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 下载失败: {url}")
            print(f"错误: {e}")
            return False
        except Exception as e:
            print(f"❌ 下载过程中出错: {e}")
            return False
    
    def extract_zip(self, zip_path: str) -> Optional[str]:
        """解压ZIP文件"""
        try:
            extract_dir = os.path.dirname(zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                if len(file_list) == 1:
                    csv_filename = file_list[0]
                    zip_ref.extractall(extract_dir)
                    csv_path = os.path.join(extract_dir, csv_filename)
                    
                    print(f"📂 解压完成: {csv_filename}")
                    
                    # 删除ZIP文件节省空间
                    os.remove(zip_path)
                    
                    return csv_path
                else:
                    print(f"⚠️  ZIP文件包含多个文件: {file_list}")
                    return None
                    
        except zipfile.BadZipFile:
            print(f"❌ 损坏的ZIP文件: {zip_path}")
            return None
        except Exception as e:
            print(f"❌ 解压失败: {e}")
            return None
    
    def process_csv(self, csv_path: str) -> pd.DataFrame:
        """处理单个CSV文件，增强时间戳处理"""
        try:
            # 币安数据列名
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                'ignore'
            ]
            
            # 读取数据
            print(f"📖 读取CSV文件: {os.path.basename(csv_path)}")
            df = pd.read_csv(csv_path, names=columns)
            print(f"📊 原始数据行数: {len(df)}")
            
            # 智能检测时间戳格式并处理
            try:
                ts_sample = df['timestamp'].iloc[:10]
                print(f"🕒 时间戳样本: {ts_sample.tolist()}")
                
                # 检测时间戳位数
                first_ts = int(df['timestamp'].iloc[0])
                ts_str = str(first_ts)
                ts_digits = len(ts_str)
                
                print(f"🔍 时间戳位数: {ts_digits}")
                
                if ts_digits == 13:
                    # 标准毫秒时间戳
                    print("✅ 检测到毫秒时间戳格式")
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                elif ts_digits == 16:
                    # 微秒时间戳（需要除以1000转换为毫秒）
                    print("✅ 检测到微秒时间戳格式，转换为毫秒")
                    df['timestamp'] = pd.to_datetime(df['timestamp'] / 1000, unit='ms')
                elif ts_digits == 10:
                    # 秒时间戳
                    print("✅ 检测到秒时间戳格式")
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                else:
                    # 尝试自动检测
                    print(f"⚠️  未知时间戳格式({ts_digits}位)，尝试自动检测...")
                    
                    # 尝试不同的单位
                    for unit in ['ms', 's']:
                        try:
                            test_ts = pd.to_datetime(df['timestamp'].iloc[0], unit=unit)
                            # 检查是否在合理时间范围内 (2010-2030)
                            if datetime(2010, 1, 1) <= test_ts <= datetime(2030, 1, 1):
                                print(f"✅ 自动检测成功，使用{unit}单位")
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit=unit)
                                break
                        except:
                            continue
                    else:
                        raise ValueError(f"无法解析时间戳格式: {ts_digits}位")
                
                # 验证转换后的时间戳
                min_time = df['timestamp'].min()
                max_time = df['timestamp'].max()
                print(f"📅 时间范围: {min_time} 到 {max_time}")
                
                # 检查时间范围是否合理 (2010-2030年)
                if min_time < datetime(2010, 1, 1) or max_time > datetime(2030, 1, 1):
                    raise ValueError(f"时间范围异常: {min_time} - {max_time}")
                
                print("✅ 时间戳转换成功")
                
            except Exception as e:
                print(f"❌ 时间戳处理失败: {e}")
                os.remove(csv_path)
                return pd.DataFrame()
            
            # 选择需要的列
            processed_df = df[['timestamp', 'open', 'high', 'low', 'close',
                              'volume']].copy()
            
            # 数据类型转换
            for col in ['open', 'high', 'low', 'close', 'volume']:
                processed_df[col] = pd.to_numeric(processed_df[col],
                                                  errors='coerce')
            
            # 检查数据质量
            null_count = processed_df.isnull().sum().sum()
            if null_count > 0:
                print(f"⚠️  发现 {null_count} 个缺失值，将进行清理")
                processed_df = processed_df.dropna()
            
            # 检查价格合理性
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (processed_df[col] <= 0).any():
                    print("⚠️  发现非正价格，将进行清理")
                    processed_df = processed_df[processed_df[col] > 0]
            
            # 检查OHLC逻辑
            invalid_ohlc = (
                (processed_df['high'] < processed_df['low']) |
                (processed_df['high'] < processed_df['open']) |
                (processed_df['high'] < processed_df['close']) |
                (processed_df['low'] > processed_df['open']) |
                (processed_df['low'] > processed_df['close'])
            )
            
            if invalid_ohlc.any():
                invalid_count = invalid_ohlc.sum()
                print(f"⚠️  发现 {invalid_count} 条OHLC逻辑错误数据，将进行清理")
                processed_df = processed_df[~invalid_ohlc]
            
            # 按时间排序
            processed_df = processed_df.sort_values('timestamp').reset_index(
                drop=True)
            
            print(f"📊 处理完成: {len(processed_df)} 条有效数据")
            
            # 删除原始CSV文件
            os.remove(csv_path)
            
            return processed_df
            
        except Exception as e:
            print(f"❌ 处理CSV失败: {e}")
            # 删除可能损坏的文件
            if os.path.exists(csv_path):
                os.remove(csv_path)
            return pd.DataFrame()
    
    def download_single_month(self, year: int, month: int) -> str:
        """下载单个月份的数据并保存"""
        print(f"\n📅 下载 {year}-{month:02d} 月数据")
        print("-" * 40)
        
        # 生成文件名和路径
        filename = f"{self.symbol}-{self.interval}-{year}-{month:02d}.zip"
        url = f"{self.base_url}/{self.symbol}/{self.interval}/{filename}"
        zip_path = os.path.join(self.data_dir, filename)
        
        # 检查月份数据是否已存在
        monthly_csv = os.path.join(self.processed_dir, f"{self.symbol}_{year}_{month:02d}.csv")
        if os.path.exists(monthly_csv):
            print(f"✅ 月份数据已存在: {monthly_csv}")
            return monthly_csv
        
        # 下载文件
        if not self.download_file(url, zip_path):
            print(f"❌ {year}-{month:02d} 下载失败")
            return ""
        
        # 解压文件
        csv_path = self.extract_zip(zip_path)
        if not csv_path:
            print(f"❌ {year}-{month:02d} 解压失败")
            return ""
        
        # 处理数据
        df = self.process_csv(csv_path)
        if df.empty:
            print(f"❌ {year}-{month:02d} 数据处理失败")
            return ""
        
        # 保存月份数据
        df.to_csv(monthly_csv, index=False)
        file_size = os.path.getsize(monthly_csv) / 1024 / 1024
        print(f"💾 {year}-{month:02d} 已保存: {monthly_csv} ({file_size:.2f}MB)")
        
        return monthly_csv
    
    def merge_monthly_files(self, file_paths: List[str], output_filename: str) -> str:
        """合并多个月份文件"""
        print(f"\n🔗 合并 {len(file_paths)} 个月份文件...")
        
        all_data = []
        total_records = 0
        
        for file_path in sorted(file_paths):
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                all_data.append(df)
                total_records += len(df)
                month_name = os.path.basename(file_path).replace('.csv', '')
                print(f"  📂 {month_name}: {len(df):,} 条记录")
        
        if not all_data:
            print("❌ 没有找到有效的月份数据")
            return ""
        
        # 合并数据
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 按时间排序
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # 去重
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        after_dedup = len(combined_df)
        
        if before_dedup != after_dedup:
            removed_count = before_dedup - after_dedup
            print(f"🧹 去重: 删除了 {removed_count} 条重复数据")
        
        # 保存合并后的数据
        output_path = os.path.join(self.processed_dir, output_filename)
        combined_df.to_csv(output_path, index=False)
        
        # 统计信息
        file_size = os.path.getsize(output_path) / 1024 / 1024
        time_span = (combined_df['timestamp'].max() - combined_df['timestamp'].min()).days
        
        print(f"\n✅ 合并完成!")
        print(f"📄 输出文件: {output_path}")
        print(f"📏 文件大小: {file_size:.2f} MB")
        print(f"📊 总记录数: {len(combined_df):,}")
        print(f"📅 时间跨度: {time_span} 天")
        print(f"🕒 时间范围: {combined_df['timestamp'].min()} 到 {combined_df['timestamp'].max()}")
        
        return output_path

    def download_month_range(self, start_year: int, start_month: int,
                           end_year: int, end_month: int) -> List[str]:
        """下载指定月份范围的数据，返回文件路径列表"""
        print(f"\n🚀 分月下载策略: {start_year}-{start_month:02d} 到 {end_year}-{end_month:02d}")
        print("=" * 60)
        
        # 生成月份列表
        months = []
        current_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 1)
        
        # 确保不会下载未来的数据
        now = datetime.now()
        max_date = datetime(now.year, now.month, 1)
        if end_date > max_date:
            end_date = max_date
            print(f"⚠️  调整结束日期到当前月份: {max_date.strftime('%Y-%m')}")
        
        while current_date <= end_date:
            months.append((current_date.year, current_date.month))
            # 下个月
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        print(f"📋 需要下载的月份: {[f'{y}-{m:02d}' for y, m in months]}")
        
        # 分月下载
        successful_files = []
        failed_months = []
        
        for i, (year, month) in enumerate(months, 1):
            print(f"\n进度: {i}/{len(months)}")
            
            try:
                monthly_file = self.download_single_month(year, month)
                if monthly_file:
                    successful_files.append(monthly_file)
                    print(f"✅ {year}-{month:02d} 下载成功")
                else:
                    failed_months.append(f"{year}-{month:02d}")
                    print(f"❌ {year}-{month:02d} 下载失败")
                
                # 添加延迟避免请求过快
                if i < len(months):  # 最后一个不需要延迟
                    time.sleep(1)
                    
            except Exception as e:
                failed_months.append(f"{year}-{month:02d}")
                print(f"❌ {year}-{month:02d} 下载异常: {e}")
        
        # 下载汇总
        print(f"\n📊 下载汇总:")
        print(f"✅ 成功: {len(successful_files)}/{len(months)} 个月份")
        print(f"❌ 失败: {len(failed_months)}/{len(months)} 个月份")
        
        if failed_months:
            print(f"❌ 失败月份: {failed_months}")
            print("💡 可以稍后重新运行程序，只下载失败的月份")
        
        return successful_files

    def quick_download_recent(self, months: int = 6) -> str:
        """智能下载最近几个月的数据，自动检查本地文件"""
        print(f"🚀 智能下载最近 {months} 个月的历史数据")
        
        # 计算需要的月份范围
        end_date = datetime.now() - timedelta(days=30)
        start_date = end_date - timedelta(days=months * 30)
        
        print(f"📅 目标时间范围: {start_date.strftime('%Y-%m')} 到 {end_date.strftime('%Y-%m')}")
        
        # 生成需要的月份列表
        required_months = []
        current_date = datetime(start_date.year, start_date.month, 1)
        end_check = datetime(end_date.year, end_date.month, 1)
        
        # 确保不超过当前时间
        now = datetime.now()
        max_date = datetime(now.year, now.month, 1)
        if end_check > max_date:
            end_check = max_date
        
        while current_date <= end_check:
            required_months.append((current_date.year, current_date.month))
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        print(f"📋 需要的月份: {[f'{y}-{m:02d}' for y, m in required_months]}")
        
        # 检查本地已有文件和需要下载的文件
        existing_files = []
        missing_months = []
        
        for year, month in required_months:
            monthly_csv = os.path.join(self.processed_dir, f"{self.symbol}_{year}_{month:02d}.csv")
            if os.path.exists(monthly_csv):
                existing_files.append(monthly_csv)
                print(f"✅ 本地已存在: {year}-{month:02d}")
            else:
                missing_months.append((year, month))
                print(f"🔍 需要下载: {year}-{month:02d}")
        
        # 下载缺失的月份
        if missing_months:
            print(f"\n📥 开始下载 {len(missing_months)} 个缺失月份...")
            download_success = 0
            
            for year, month in missing_months:
                try:
                    monthly_file = self.download_single_month(year, month)
                    if monthly_file:
                        existing_files.append(monthly_file)
                        download_success += 1
                        print(f"✅ {year}-{month:02d} 下载成功")
                    else:
                        print(f"❌ {year}-{month:02d} 下载失败")
                    
                    # 添加延迟避免请求过快
                    time.sleep(1)
                except Exception as e:
                    print(f"❌ {year}-{month:02d} 下载异常: {e}")
            
            print(f"\n📊 下载结果: {download_success}/{len(missing_months)} 个月份成功")
        else:
            print("✅ 所有需要的月份文件都已存在本地")
        
        # 检查是否获得了所有需要的文件
        if len(existing_files) < len(required_months):
            missing_count = len(required_months) - len(existing_files)
            print(f"⚠️  仍有 {missing_count} 个月份缺失，将使用已有的 {len(existing_files)} 个月份数据")
        
        if not existing_files:
            print("❌ 没有任何可用的月份数据")
            return ""
        
        # 自动合并所有可用的月份数据
        print(f"\n🔗 自动合并 {len(existing_files)} 个月份文件...")
        output_filename = f"{self.symbol}_recent_{months}months.csv"
        merged_file = self.merge_monthly_files(existing_files, output_filename)
        
        if merged_file:
            print(f"\n🎉 完成！最终数据文件: {merged_file}")
            if len(existing_files) == len(required_months):
                print("✅ 所有月份数据完整")
            else:
                print(f"⚠️  使用了 {len(existing_files)}/{len(required_months)} 个月份的数据")
        
        return merged_file


def main():
    """主函数 - 提供几种使用方式"""
    print("📊 币安数据下载器 - 分月下载策略")
    print("=" * 50)
    
    # 创建下载器
    downloader = BinanceDataDownloader("USDCUSDT", "1m")
    
    # 检查已存在的月份文件
    existing_files = []
    processed_dir = downloader.processed_dir
    if os.path.exists(processed_dir):
        for file in os.listdir(processed_dir):
            if file.startswith("USDCUSDT_") and file.endswith(".csv") and "_recent_" not in file:
                existing_files.append(file)
    
    if existing_files:
        print(f"\n📂 发现已存在的月份数据文件：")
        for file in sorted(existing_files):
            file_path = os.path.join(processed_dir, file)
            file_size = os.path.getsize(file_path) / 1024 / 1024
            print(f"  📄 {file} ({file_size:.2f}MB)")
    
    print("\n请选择下载方式:")
    print("1. 快速下载最近6个月历史数据（分月策略）")
    print("2. 快速下载最近12个月历史数据")
    print("3. 自定义时间范围下载")
    print("4. 重新下载失败的月份（4月）")
    print("5. 合并现有月份文件")
    print("6. 查看已下载的月份状态")
    
    try:
        choice = input("\n请输入选择 (1-6): ").strip()
        
        if choice == "1":
            # 最近6个月历史数据
            save_path = downloader.quick_download_recent(6)
            if save_path:
                print(f"\n🎉 下载完成！数据文件: {save_path}")
                print("\n💡 使用提示:")
                print("在深度学习程序中修改配置:")
                print(f"data_file: str = \"{save_path}\"")
            
        elif choice == "2":
            # 最近12个月历史数据
            save_path = downloader.quick_download_recent(12)
            if save_path:
                print(f"\n🎉 下载完成！数据文件: {save_path}")
        
        elif choice == "3":
            # 自定义范围
            print("\n请输入开始时间:")
            start_year = int(input("开始年份 (如 2024): "))
            start_month = int(input("开始月份 (1-12): "))
            
            print("\n请输入结束时间:")
            end_year = int(input("结束年份 (如 2025): "))
            end_month = int(input("结束月份 (1-12): "))
            
            monthly_files = downloader.download_month_range(start_year, start_month, end_year, end_month)
            if monthly_files:
                filename = f"USDCUSDT_{start_year}_{start_month:02d}_to_{end_year}_{end_month:02d}.csv"
                merged_file = downloader.merge_monthly_files(monthly_files, filename)
                if merged_file:
                    print(f"\n🎉 下载完成！数据文件: {merged_file}")
        
        elif choice == "4":
            # 重新下载失败的月份（特别是4月）
            print("\n🔄 重新下载2025年4月数据...")
            monthly_file = downloader.download_single_month(2025, 4)
            if monthly_file:
                print(f"✅ 4月数据下载成功: {monthly_file}")
                
                # 询问是否重新合并6个月数据
                merge_choice = input("\n是否重新合并完整的6个月数据？(y/n): ").strip().lower()
                if merge_choice == 'y':
                    # 查找所有月份文件
                    months_files = []
                    for year, month in [(2024, 11), (2024, 12), (2025, 1), (2025, 2), (2025, 3), (2025, 4), (2025, 5)]:
                        file_path = os.path.join(processed_dir, f"USDCUSDT_{year}_{month:02d}.csv")
                        if os.path.exists(file_path):
                            months_files.append(file_path)
                    
                    if len(months_files) >= 6:  # 至少6个月
                        merged_file = downloader.merge_monthly_files(months_files, "USDCUSDT_recent_6months.csv")
                        if merged_file:
                            print(f"\n🎉 重新合并完成！数据文件: {merged_file}")
            else:
                print("❌ 4月数据下载失败")
        
        elif choice == "5":
            # 合并现有月份文件
            if not existing_files:
                print("❌ 没有找到可合并的月份文件")
                return
            
            print(f"\n📂 找到 {len(existing_files)} 个月份文件")
            merge_choice = input("是否合并所有月份文件？(y/n): ").strip().lower()
            if merge_choice == 'y':
                file_paths = [os.path.join(processed_dir, f) for f in existing_files]
                merged_file = downloader.merge_monthly_files(file_paths, "USDCUSDT_merged_all.csv")
                if merged_file:
                    print(f"\n🎉 合并完成！数据文件: {merged_file}")
        
        elif choice == "6":
            # 查看已下载的月份状态
            print("\n📊 已下载月份状态:")
            if not existing_files:
                print("❌ 没有找到任何月份数据文件")
            else:
                total_size = 0
                total_records = 0
                for file in sorted(existing_files):
                    file_path = os.path.join(processed_dir, file)
                    file_size = os.path.getsize(file_path) / 1024 / 1024
                    total_size += file_size
                    
                    # 读取记录数
                    try:
                        df = pd.read_csv(file_path)
                        records = len(df)
                        total_records += records
                        print(f"  📄 {file}: {records:,} 条记录 ({file_size:.2f}MB)")
                    except:
                        print(f"  📄 {file}: 无法读取 ({file_size:.2f}MB)")
                
                print(f"\n📊 汇总:")
                print(f"  总文件数: {len(existing_files)}")
                print(f"  总大小: {total_size:.2f}MB")
                print(f"  总记录数: {total_records:,}")
        
        else:
            print("❌ 无效选择")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  用户取消下载")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")


if __name__ == "__main__":
    main() 