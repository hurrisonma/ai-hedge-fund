#!/usr/bin/env python3
"""
🔄 aggTrades特征工程模块
专门处理币安aggTrades数据，为稳定币预测优化

📊 数据基准备忘录：
═══════════════════════════════════════════════════════════════════

🎯 核心设计理念：
- 使用best_bid价格作为close价格基准
- 基于流动性特征进行稳定币价格预测
- 放弃传统技术指标（SMA/EMA/RSI等），专注aggTrades特征

💰 价格基准详情：
- data['close'] = data['best_bid']  # 买一价作为收盘价
- 预测目标：未来N分钟后的bid价格变化方向
- 变化率计算：(future_bid - current_bid) / current_bid
- 分类阈值：±0.0001 (±0.01%)

🔧 特征组成（共25个）：
1. 市场深度特征(5个)：
   - bid_depth_1, ask_depth_1, depth_imbalance, total_depth, depth_ratio
2. 价格特征(2个)：best_bid, best_ask（移除spread相关）
3. 成交量特征(4个)：total_volume, buy_volume, sell_volume, buy_ratio
4. 流动性特征(4个)：trade_count, avg_trade_size, large_trade_count, large_trade_ratio
5. 市场冲击特征(2个)：price_range, volume_impact
6. 时间特征(8个)：
   - 周期编码：hour_sin/cos, day_of_week_sin/cos, day_of_month_sin/cos
   - 市场状态：is_market_open, session_type

📈 标签生成逻辑：
- 时间点预测（非时间段）
- labels = np.where(returns > 0.0001, 0,      # 上涨
                   np.where(returns < -0.0001, 1,   # 下跌
                          2))                       # 持平

⚖️ 类别分布特征：
- 持平类占95%+（USDCUSDT稳定币特性）
- 上涨/下跌类稀少但重要
- 极不平衡，需要特殊处理策略

最后更新：2025-06-02
═══════════════════════════════════════════════════════════════════
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd


class AggTradesFeatureEngineer:
    """aggTrades数据特征工程器"""
    
    def __init__(self):
        self.feature_names = []
        
    def process_aggtrades_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理aggTrades数据，添加必要特征
        
        Args:
            df: aggTrades原始数据
            
        Returns:
            处理后的特征数据
        """
        print("🔄 处理aggTrades数据...")
        
        data = df.copy()
        
        # 1. 添加close价格（使用bid价格作为基准）
        data['close'] = data['best_bid']
        print("✅ 使用best_bid作为close价格基准")
        
        # 2. 计算市场深度特征（基于成交量推算）
        data = self._add_depth_features(data)
        
        # 3. 构建aggTrades核心特征列表（移除spread相关，添加深度特征）
        aggtrades_features = [
            # 市场深度特征（新增5个）
            'bid_depth_1', 'ask_depth_1', 'depth_imbalance', 'total_depth', 'depth_ratio',
            # 价格特征（保留2个，移除spread相关）
            'best_bid', 'best_ask',
            # 成交量特征（保留4个）
            'total_volume', 'buy_volume', 'sell_volume', 'buy_ratio',
            # 流动性特征（保留4个）
            'trade_count', 'avg_trade_size', 'large_trade_count', 'large_trade_ratio',
            # 市场冲击特征（保留2个）
            'price_range', 'volume_impact'
        ]
        
        self.feature_names = aggtrades_features.copy()
        
        # 4. 添加时间特征
        data = self._add_time_features(data)
        
        # 5. 数据清理
        data = self._clean_features(data)
        
        print(f"✅ 特征工程完成，特征维度: {len(self.feature_names)}")
        return data
    
    def _add_depth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加市场深度特征（基于成交量推算）"""
        print("  📊 添加市场深度特征...")
        
        # 使用买卖成交量作为深度的代理指标
        df['bid_depth_1'] = df['buy_volume']     # 买方深度 = 买单成交量
        df['ask_depth_1'] = df['sell_volume']    # 卖方深度 = 卖单成交量
        
        # 计算深度不平衡度
        total_depth = df['bid_depth_1'] + df['ask_depth_1']
        df['depth_imbalance'] = np.where(
            total_depth > 0,
            (df['bid_depth_1'] - df['ask_depth_1']) / total_depth,
            0.0
        )
        
        # 总深度
        df['total_depth'] = total_depth
        
        # 深度比例（买方深度/卖方深度）
        df['depth_ratio'] = np.where(
            df['ask_depth_1'] > 0,
            df['bid_depth_1'] / df['ask_depth_1'],
            1.0
        )
        
        print("    ✅ 深度特征计算完成")
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间周期特征"""
        print("  🕒 添加时间特征...")
        
        if 'timestamp' not in df.columns:
            print("    ⚠️ 无时间戳列，跳过时间特征")
            # 创建占位符时间特征
            for feature in ['hour_sin', 'hour_cos', 'day_of_week_sin', 
                          'day_of_week_cos', 'day_of_month_sin', 
                          'day_of_month_cos', 'is_market_open', 'session_type']:
                df[feature] = 0.0
        else:
            # 确保timestamp是datetime类型
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 小时的周期性编码
            hour = df['timestamp'].dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            # 星期的周期性编码
            day_of_week = df['timestamp'].dt.dayofweek
            df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            
            # 月内日期的周期性编码
            day_of_month = df['timestamp'].dt.day
            df['day_of_month_sin'] = np.sin(2 * np.pi * day_of_month / 31)
            df['day_of_month_cos'] = np.cos(2 * np.pi * day_of_month / 31)
            
            # 市场时间特征（24小时交易）
            df['is_market_open'] = 1.0
            
            # 交易时段分类
            session_map = {
                range(0, 8): 0,    # 亚洲时段
                range(8, 16): 1,   # 欧洲时段
                range(16, 24): 2   # 美洲时段
            }
            df['session_type'] = hour.apply(
                lambda h: next((v for k, v in session_map.items() 
                              if h in k), 0)
            )
        
        # 添加时间特征到特征名列表
        time_features = [
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos', 'is_market_open', 
            'session_type'
        ]
        self.feature_names.extend(time_features)
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理和标准化特征"""
        print("  🧹 清理特征数据...")
        
        # 获取特征列
        feature_cols = [col for col in self.feature_names if col in df.columns]
        
        # 处理缺失值（前向填充）
        df[feature_cols] = df[feature_cols].fillna(method='ffill')
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # 处理无穷值
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
        
        # 移除异常值（3σ原则，跳过分类特征）
        categorical_features = ['is_market_open', 'session_type']
        for col in feature_cols:
            if col not in categorical_features:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:  # 避免除零
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val
                    df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def create_labels(self, df: pd.DataFrame, horizons: List[int], 
                     threshold: float = 0.0001, two_task_mode: bool = False) -> pd.DataFrame:
        """
        创建预测标签 - 专注稳定性检测
        
        Args:
            df: 包含close价格的DataFrame
            horizons: 预测时间范围（分钟）
            threshold: 价格变化阈值
            two_task_mode: 是否使用两任务模式（现在默认False）
            
        Returns:
            包含标签的DataFrame
        """
        print(f"🎯 创建预测标签（稳定性检测），时间范围: {horizons}分钟")
        
        data = df.copy()
        
        for horizon in horizons:
            # 计算未来收益率（基于bid价格）
            future_price = data['close'].shift(-horizon)
            current_price = data['close']
            returns = (future_price - current_price) / current_price
            
            if two_task_mode:
                # 两任务模式（保留兼容性，但默认不使用）
                # ... existing two_task_mode code ...
                pass
            else:
                # 🔥 单任务模式：稳定性检测（0=稳定, 1=偏离）
                is_stable = (returns.abs() <= threshold)
                stability_labels = (~is_stable).astype(int)  # 稳定=0, 偏离=1
                
                data[f'label_{horizon}min'] = stability_labels
                
                # 统计标签分布
                total_count = len(stability_labels)
                stable_count = is_stable.sum()
                unstable_count = (~is_stable).sum()
                
                print(f"  {horizon}分钟稳定性标签分布:")
                print(f"    稳定(0)={stable_count} ({stable_count/total_count*100:.1f}%)")
                print(f"    偏离(1)={unstable_count} ({unstable_count/total_count*100:.1f}%)")
        
        # 移除无法预测的末尾数据
        max_horizon = max(horizons)
        data = data[:-max_horizon].copy()
        
        print(f"✅ 稳定性标签创建完成，有效样本数: {len(data)}")
        return data
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        return self.feature_names.copy()
    
    def feature_summary(self, df: pd.DataFrame) -> Dict:
        """特征摘要统计"""
        feature_cols = [col for col in self.feature_names if col in df.columns]
        
        summary = {
            'total_features': len(feature_cols),
            'feature_groups': {
                'aggtrades_core': 17,
                'time_features': 8
            },
            'data_shape': df.shape,
            'missing_values': df[feature_cols].isnull().sum().sum(),
            'price_range': {
                'bid_min': df['best_bid'].min(),
                'bid_max': df['best_bid'].max(),
                'bid_mean': df['best_bid'].mean()
            },
            'volume_stats': {
                'total_volume_mean': df['total_volume'].mean(),
                'buy_ratio_mean': df['buy_ratio'].mean(),
                'trade_count_mean': df['trade_count'].mean()
            }
        }
        
        return summary


def test_aggtrades_features():
    """测试aggTrades特征工程功能"""
    print("🧪 测试aggTrades特征工程模块...")
    
    # 加载真实aggTrades数据
    try:
        df = pd.read_csv('data/processed/USDCUSDT_aggTrades_recent_6months.csv')
        print(f"✅ 加载数据成功: {df.shape}")
        
        # 测试特征工程
        fe = AggTradesFeatureEngineer()
        features_df = fe.process_aggtrades_data(df.head(1000))  # 测试前1000行
        labels_df = fe.create_labels(features_df, [5, 10, 15])
        
        # 打印结果
        print(f"原始数据形状: {df.shape}")
        print(f"特征数据形状: {features_df.shape}")
        print(f"带标签数据形状: {labels_df.shape}")
        print(f"特征名称({len(fe.get_feature_names())}个): {fe.get_feature_names()}")
        
        # 特征摘要
        summary = fe.feature_summary(features_df)
        print(f"特征摘要: {summary}")
        
        return features_df, labels_df
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None, None


if __name__ == "__main__":
    test_aggtrades_features() 