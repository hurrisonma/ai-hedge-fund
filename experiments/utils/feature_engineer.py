"""
🧪 特征工程模块
独立实验程序，基于K线数据生成深度学习特征
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple
from datetime import datetime

warnings.filterwarnings('ignore')

class FeatureEngineer:
    """K线数据特征工程师"""
    
    def __init__(self):
        self.feature_names = []
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从K线数据创建所有特征
        
        Args:
            df: 包含OHLCV的DataFrame
            
        Returns:
            包含所有特征的DataFrame
        """
        print("🔧 开始特征工程...")
        
        # 复制数据避免修改原始数据
        data = df.copy()
        
        # 确保数据按时间排序
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp').reset_index(drop=True)
        
        # 1. 基础价格特征
        data = self._add_basic_features(data)
        
        # 2. 价格衍生特征
        data = self._add_price_derived_features(data)
        
        # 3. 技术指标特征
        data = self._add_technical_indicators(data)
        
        # 4. 时间特征
        data = self._add_time_features(data)
        
        # 5. 清理和标准化
        data = self._clean_features(data)
        
        print(f"✅ 特征工程完成，总特征数: {len(self.feature_names)}")
        return data
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加基础OHLCV特征"""
        print("  📊 添加基础特征...")
        
        # 确保基础列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必需列: {col}")
        
        # 基础特征已经存在，只需要添加到特征名单
        basic_features = ['open', 'high', 'low', 'close', 'volume']
        self.feature_names.extend(basic_features)
        
        return df
    
    def _add_price_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格衍生特征"""
        print("  💰 添加价格衍生特征...")
        
        # 价格变化特征
        df['price_change'] = (df['close'] - df['open']) / df['open']
        df['high_low_range'] = (df['high'] - df['low']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        df['body_size'] = np.abs(df['close'] - df['open']) / df['open']
        
        # K线类型
        df['is_green'] = (df['close'] > df['open']).astype(int)
        
        # 成交量相关
        df['volume_price_ratio'] = df['volume'] / df['close']
        df['volatility'] = (df['high'] - df['low']) / df['close']
        
        # 动量特征（需要前一根K线）
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['momentum'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        
        price_derived_features = [
            'price_change', 'high_low_range', 'upper_shadow', 'lower_shadow', 'body_size',
            'is_green', 'volume_price_ratio', 'volatility', 'gap', 'momentum'
        ]
        self.feature_names.extend(price_derived_features)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标特征"""
        print("  📈 添加技术指标...")
        
        # 移动平均线
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # 指数移动平均
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # RSI指标
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # MACD指标
        macd_line, macd_signal, macd_hist = self._calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # 布林带
        bb_upper, bb_lower, bb_width = self._calculate_bollinger_bands(df['close'], 20, 2)
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_width'] = bb_width
        
        # 成交量指标
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        
        technical_features = [
            'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_lower', 'bb_width', 'volume_sma_10', 'volume_ratio'
        ]
        self.feature_names.extend(technical_features)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        print("  🕐 添加时间特征...")
        
        if 'timestamp' not in df.columns:
            print("    ⚠️ 无时间戳列，跳过时间特征")
            # 创建占位符时间特征
            for feature in ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                          'day_of_month_sin', 'day_of_month_cos', 'is_market_open', 'session_type']:
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
            
            # 市场时间特征（假设24小时交易）
            df['is_market_open'] = 1.0  # 假设总是开市
            
            # 交易时段（简化版本）
            session_map = {
                range(0, 8): 0,    # 亚洲时段
                range(8, 16): 1,   # 欧洲时段
                range(16, 24): 2   # 美洲时段
            }
            df['session_type'] = hour.apply(
                lambda h: next((v for k, v in session_map.items() if h in k), 0)
            )
        
        time_features = [
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos', 'is_market_open', 'session_type'
        ]
        self.feature_names.extend(time_features)
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理和标准化特征"""
        print("  🧹 清理特征数据...")
        
        # 处理缺失值（前向填充）
        feature_cols = [col for col in self.feature_names if col in df.columns]
        df[feature_cols] = df[feature_cols].fillna(method='ffill')
        df[feature_cols] = df[feature_cols].fillna(0)  # 如果还有NaN，填0
        
        # 处理无穷值
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
        
        # 移除异常值（3σ原则）
        for col in feature_cols:
            if col not in ['is_green', 'is_market_open', 'session_type']:  # 跳过分类特征
                mean_val = df[col].mean()
                std_val = df[col].std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_hist = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        width = upper_band - lower_band
        
        return upper_band, lower_band, width
    
    def create_labels(self, df: pd.DataFrame, horizons: List[int], threshold: float = 0.001) -> pd.DataFrame:
        """
        创建预测标签
        
        Args:
            df: 包含close价格的DataFrame
            horizons: 预测时间范围（分钟）
            threshold: 价格变化阈值
            
        Returns:
            包含标签的DataFrame
        """
        print(f"🎯 创建预测标签，时间范围: {horizons}分钟...")
        
        data = df.copy()
        
        for horizon in horizons:
            # 计算未来收益率
            future_price = data['close'].shift(-horizon)
            current_price = data['close']
            returns = (future_price - current_price) / current_price
            
            # 分类标签
            labels = np.where(returns > threshold, 0,     # 上涨
                            np.where(returns < -threshold, 1,  # 下跌
                                   2))                          # 持平
            
            data[f'label_{horizon}min'] = labels
        
        # 移除无法预测的末尾数据
        max_horizon = max(horizons)
        data = data[:-max_horizon].copy()
        
        print(f"✅ 标签创建完成，有效样本数: {len(data)}")
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
                'basic': 5,
                'price_derived': 10,
                'technical': 15,
                'time': 8
            },
            'data_shape': df.shape,
            'missing_values': df[feature_cols].isnull().sum().sum(),
            'feature_stats': df[feature_cols].describe()
        }
        
        return summary

def test_feature_engineer():
    """测试特征工程功能"""
    print("🧪 测试特征工程模块...")
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    base_price = 1.0
    
    mock_data = pd.DataFrame({
        'timestamp': dates,
        'open': base_price + np.random.randn(n_samples) * 0.001,
        'high': base_price + np.random.randn(n_samples) * 0.001 + 0.0005,
        'low': base_price + np.random.randn(n_samples) * 0.001 - 0.0005,
        'close': base_price + np.random.randn(n_samples) * 0.001,
        'volume': np.random.randint(10000, 100000, n_samples)
    })
    
    # 确保OHLC关系正确
    mock_data['high'] = mock_data[['open', 'close']].max(axis=1) + np.abs(np.random.randn(n_samples) * 0.0002)
    mock_data['low'] = mock_data[['open', 'close']].min(axis=1) - np.abs(np.random.randn(n_samples) * 0.0002)
    
    # 测试特征工程
    fe = FeatureEngineer()
    features_df = fe.create_all_features(mock_data)
    labels_df = fe.create_labels(features_df, [5, 10, 15])
    
    # 打印结果
    print(f"原始数据形状: {mock_data.shape}")
    print(f"特征数据形状: {features_df.shape}")
    print(f"带标签数据形状: {labels_df.shape}")
    print(f"特征名称: {fe.get_feature_names()}")
    
    return features_df, labels_df

if __name__ == "__main__":
    test_feature_engineer() 