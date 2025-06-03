"""
🧪 数据加载器
独立实验程序，处理aggTrades数据，创建时序序列

📊 数据基准备忘录：
═══════════════════════════════════════════════════════════════════

🎯 当前配置：
- 数据源：aggTrades聚合数据（bid价格基准）
- 特征工程：AggTradesFeatureEngineer
- 序列长度：30分钟历史窗口
- 预测目标：5/10/15分钟后的bid价格变化方向

💰 价格基准：
- 使用best_bid作为close价格
- 预测未来时刻的价格变化（非时间段）
- 分类阈值：±0.01%

🔧 数据流程：
1. 加载aggTrades CSV数据
2. 使用AggTradesFeatureEngineer处理特征
3. 创建30分钟滑动窗口序列
4. 标准化特征（保持时序一致性）
5. 生成PyTorch DataLoader

⚖️ 类别平衡处理：
- 极不平衡数据（持平95%+）
- 需要适当的采样策略
- 考虑加权损失函数

最后更新：2025-06-02
═══════════════════════════════════════════════════════════════════
"""

import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

class KLineDataset(Dataset):
    """K线时序数据集"""
    
    def __init__(self, features: np.ndarray, labels: Dict[str, np.ndarray], 
                 sequence_length: int, transform: Optional[callable] = None):
        """
        初始化数据集
        
        Args:
            features: 特征数组 [n_samples, n_features]
            labels: 标签字典 {horizon: [n_samples]}
            sequence_length: 序列长度
            transform: 数据变换函数
        """
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform
        
        # 计算有效样本数量（需要足够的历史数据）
        total_samples = len(features) - sequence_length + 1
        
        # 🔧 过滤掉无效标签（-1）的样本
        self.valid_indices = []
        for idx in range(total_samples):
            label_idx = idx + sequence_length - 1
            # 检查所有时间尺度的标签是否都有效
            all_valid = True
            for horizon, label_array in labels.items():
                if label_idx < len(label_array) and label_array[label_idx] == -1:
                    all_valid = False
                    break
            if all_valid:
                self.valid_indices.append(idx)
        
        self.n_samples = len(self.valid_indices)
        self.feature_dim = features.shape[1]
        
        print(f"  📊 数据集统计: 总样本={total_samples}, 有效样本={self.n_samples} ({self.n_samples/total_samples*100:.1f}%)")
        
        # 标签对齐检查
        for horizon, label_array in labels.items():
            if len(label_array) != len(features):
                raise ValueError(f"特征和标签{horizon}长度不匹配: {len(features)} vs {len(label_array)}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (features, labels) 元组
        """
        if idx >= self.n_samples:
            raise IndexError(f"索引{idx}超出范围{self.n_samples}")
        
        # 获取有效样本的真实索引
        real_idx = self.valid_indices[idx]
        
        # 获取序列特征 [sequence_length, feature_dim]
        start_idx = real_idx
        end_idx = real_idx + self.sequence_length
        sequence_features = self.features[start_idx:end_idx]
        
        # 获取对应的标签（使用序列最后一个时间点的标签）
        label_idx = end_idx - 1
        sample_labels = {}
        for horizon, label_array in self.labels.items():
            if label_idx < len(label_array):
                label_value = label_array[label_idx]
                # 确保不是无效标签
                if label_value == -1:
                    raise ValueError(f"遇到无效标签，索引={idx}, 标签索引={label_idx}")
                sample_labels[horizon] = label_value
            else:
                # 如果标签不足，这不应该发生（因为我们已经过滤了）
                raise ValueError(f"标签索引超出范围：{label_idx} >= {len(label_array)}")
        
        # 转换为tensor
        sequence_tensor = torch.FloatTensor(sequence_features)
        label_tensors = {k: torch.LongTensor([v]) for k, v in sample_labels.items()}
        
        # 应用变换
        if self.transform:
            sequence_tensor = self.transform(sequence_tensor)
        
        return sequence_tensor, label_tensors

class KLineDataProcessor:
    """K线数据处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_scaler = None
        self.feature_names = []
        self.label_encodings = {}
        
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """
        加载CSV数据
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            原始数据DataFrame
        """
        print(f"📂 加载CSV数据: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据文件不存在: {csv_path}")
        
        try:
            # 尝试自动检测CSV格式
            df = pd.read_csv(csv_path)
            print(f"  ✅ 成功加载 {len(df)} 行数据")
            print(f"  📊 数据列: {list(df.columns)}")
            
            # 检查aggTrades必需列
            required_cols = ['timestamp', 'best_bid', 'best_ask', 'total_volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  ⚠️ 缺少列: {missing_cols}")
                # 尝试映射列名
                df = self._map_column_names(df)
            
            return df
            
        except Exception as e:
            raise ValueError(f"CSV文件加载失败: {e}")
    
    def _map_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """映射列名到标准格式"""
        column_mapping = self.config.get('csv_columns', {})
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            print(f"  🔄 列名映射完成: {column_mapping}")
        
        return df
    
    def prepare_features_and_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
        """
        准备特征和标签
        
        Args:
            df: 包含特征和标签的DataFrame
            
        Returns:
            (features, labels, feature_names)
        """
        # 使用新的AggTradesFeatureEngineer
        import os
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.aggtrades_features import AggTradesFeatureEngineer
        
        print("🔧 准备aggTrades特征和标签...")
        
        # 使用aggTrades特征工程
        fe = AggTradesFeatureEngineer()
        features_df = fe.process_aggtrades_data(df)
        
        # 创建标签
        horizons = self.config.get('prediction_horizons', [5, 10, 15])
        threshold = self.config.get('price_change_threshold', 0.0001)
        use_two_task = self.config.get('use_two_task_mode', False)
        labels_df = fe.create_labels(features_df, horizons, threshold, two_task_mode=use_two_task)
        
        # 提取特征矩阵
        feature_names = fe.get_feature_names()
        features = labels_df[feature_names].values
        
        # 提取标签（支持两任务模式）
        labels = {}
        if use_two_task:
            # 两任务模式：分别提取稳定性和方向标签
            for horizon in horizons:
                stability_col = f'stability_{horizon}min'
                direction_col = f'direction_{horizon}min'
                if stability_col in labels_df.columns:
                    labels[f'stability_{horizon}min'] = labels_df[stability_col].values
                if direction_col in labels_df.columns:
                    labels[f'direction_{horizon}min'] = labels_df[direction_col].values
        else:
            # 传统模式：三分类标签
            for horizon in horizons:
                label_col = f'label_{horizon}min'
                if label_col in labels_df.columns:
                    labels[f'{horizon}min'] = labels_df[label_col].values
        
        print(f"  ✅ 特征形状: {features.shape}")
        print(f"  ✅ 标签数量: {len(labels)}")
        for horizon, label_array in labels.items():
            # 🔧 修复：使用pandas统计，可以处理-1标签
            label_counts = pd.Series(label_array).value_counts().sort_index()
            print(f"    {horizon}: {label_array.shape}, 类别分布: {label_counts.to_dict()}")
        
        return features, labels, feature_names
    
    def split_data(self, features: np.ndarray, labels: Dict[str, np.ndarray]) -> Tuple[
        Tuple[np.ndarray, Dict[str, np.ndarray]],  # train
        Tuple[np.ndarray, Dict[str, np.ndarray]],  # val  
        Tuple[np.ndarray, Dict[str, np.ndarray]]   # test
    ]:
        """
        时序数据分割（按时间顺序）
        
        Args:
            features: 特征数组
            labels: 标签字典
            
        Returns:
            (train, val, test) 数据元组
        """
        print("📊 分割训练/验证/测试集...")
        
        n_samples = len(features)
        train_ratio = self.config.get('train_ratio', 0.8)
        val_ratio = self.config.get('val_ratio', 0.1)
        
        # 计算分割点
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # 分割特征
        train_features = features[:train_end]
        val_features = features[train_end:val_end]
        test_features = features[val_end:]
        
        # 分割标签
        train_labels = {k: v[:train_end] for k, v in labels.items()}
        val_labels = {k: v[train_end:val_end] for k, v in labels.items()}
        test_labels = {k: v[val_end:] for k, v in labels.items()}
        
        print(f"  📈 训练集: {len(train_features)} 样本")
        print(f"  📊 验证集: {len(val_features)} 样本")
        print(f"  📋 测试集: {len(test_features)} 样本")
        
        return (train_features, train_labels), (val_features, val_labels), (test_features, test_labels)
    
    def fit_scaler(self, train_features: np.ndarray) -> None:
        """
        拟合特征缩放器
        
        Args:
            train_features: 训练集特征
        """
        print("📏 拟合特征缩放器...")
        
        # 使用RobustScaler（对异常值更鲁棒）
        self.feature_scaler = RobustScaler()
        
        # 只在训练数据上拟合
        self.feature_scaler.fit(train_features)
        
        print("  ✅ 缩放器拟合完成")
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        应用特征缩放
        
        Args:
            features: 原始特征
            
        Returns:
            缩放后的特征
        """
        if self.feature_scaler is None:
            raise ValueError("缩放器未拟合，请先调用fit_scaler()")
        
        return self.feature_scaler.transform(features)
    
    def create_data_loaders(self, train_data: Tuple, val_data: Tuple, test_data: Tuple) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建数据加载器
        
        Args:
            train_data: 训练数据 (features, labels)
            val_data: 验证数据 (features, labels)
            test_data: 测试数据 (features, labels)
            
        Returns:
            (train_loader, val_loader, test_loader)
        """
        print("🔄 创建数据加载器...")
        
        sequence_length = self.config.get('sequence_length', 30)
        batch_size = self.config.get('batch_size', 64)
        
        # 创建数据集
        train_dataset = KLineDataset(train_data[0], train_data[1], sequence_length)
        val_dataset = KLineDataset(val_data[0], val_data[1], sequence_length)
        test_dataset = KLineDataset(test_data[0], test_data[1], sequence_length)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  # 训练时随机打乱
            num_workers=0,  # 避免多进程问题
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证时保持顺序
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size, 
            shuffle=False,  # 测试时保持顺序
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"  ✅ 训练批次数: {len(train_loader)}")
        print(f"  ✅ 验证批次数: {len(val_loader)}")
        print(f"  ✅ 测试批次数: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def save_preprocessor(self, save_path: str) -> None:
        """保存预处理器状态"""
        preprocessor_state = {
            'feature_scaler': self.feature_scaler,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessor_state, f)
        
        print(f"💾 预处理器状态已保存: {save_path}")
    
    def load_preprocessor(self, load_path: str) -> None:
        """加载预处理器状态"""
        with open(load_path, 'rb') as f:
            preprocessor_state = pickle.load(f)
        
        self.feature_scaler = preprocessor_state['feature_scaler']
        self.feature_names = preprocessor_state['feature_names']
        
        print(f"📂 预处理器状态已加载: {load_path}")

def create_sample_data():
    """创建示例数据文件（用于测试）"""
    print("🧪 创建示例K线数据...")
    
    # 生成1年的分钟级数据
    np.random.seed(42)
    n_samples = 365 * 24 * 60  # 1年的分钟数
    
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    # 生成逼真的OHLCV数据
    base_price = 1.0856
    prices = []
    volumes = []
    
    current_price = base_price
    for i in range(n_samples):
        # 价格随机游走
        price_change = np.random.randn() * 0.0001
        current_price += price_change
        
        # 确保价格在合理范围内
        current_price = max(0.5, min(2.0, current_price))
        
        # 生成OHLC
        high_offset = abs(np.random.randn()) * 0.0002
        low_offset = abs(np.random.randn()) * 0.0002
        close_change = np.random.randn() * 0.0001
        
        open_price = current_price
        high_price = current_price + high_offset
        low_price = current_price - low_offset
        close_price = current_price + close_change
        
        # 确保OHLC关系正确
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        prices.append([open_price, high_price, low_price, close_price])
        
        # 生成成交量
        volume = np.random.randint(10000, 100000)
        volumes.append(volume)
        
        current_price = close_price
    
    # 创建DataFrame
    prices = np.array(prices)
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices[:, 0],
        'high': prices[:, 1], 
        'low': prices[:, 2],
        'close': prices[:, 3],
        'volume': volumes
    })
    
    # 保存到文件
    os.makedirs('experiments/data/raw', exist_ok=True)
    sample_path = 'experiments/data/raw/sample_kline_1year.csv'
    sample_data.to_csv(sample_path, index=False)
    
    print(f"✅ 示例数据已创建: {sample_path}")
    print(f"📊 数据形状: {sample_data.shape}")
    print(f"📈 价格范围: {sample_data['close'].min():.4f} - {sample_data['close'].max():.4f}")
    
    return sample_path

def test_data_loader():
    """测试数据加载器"""
    print("🧪 测试数据加载器...")
    
    # 创建示例数据
    sample_path = create_sample_data()
    
    # 配置
    config = {
        'data_file': sample_path,
        'sequence_length': 30,
        'prediction_horizons': [5, 10, 15],
        'price_change_threshold': 0.001,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'batch_size': 64
    }
    
    # 创建数据处理器
    processor = KLineDataProcessor(config)
    
    # 加载和处理数据
    df = processor.load_csv_data(sample_path)
    features, labels, feature_names = processor.prepare_features_and_labels(df)
    
    # 分割数据
    train_data, val_data, test_data = processor.split_data(features, labels)
    
    # 拟合缩放器
    processor.fit_scaler(train_data[0])
    
    # 应用缩放
    train_features = processor.transform_features(train_data[0])
    val_features = processor.transform_features(val_data[0])
    test_features = processor.transform_features(test_data[0])
    
    train_data = (train_features, train_data[1])
    val_data = (val_features, val_data[1])
    test_data = (test_features, test_data[1])
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        train_data, val_data, test_data
    )
    
    # 测试一个批次
    for batch_features, batch_labels in train_loader:
        print(f"批次特征形状: {batch_features.shape}")
        print(f"批次标签:")
        for horizon, labels_tensor in batch_labels.items():
            print(f"  {horizon}: {labels_tensor.shape}")
        break
    
    print("✅ 数据加载器测试完成")
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    test_data_loader() 