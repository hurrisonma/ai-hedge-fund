"""
🧪 实验配置文件
独立实验程序，不影响现有工程

📊 当前数据基准备忘录：
═══════════════════════════════════════════════════════════════════

🎯 数据源：币安aggTrades数据
- 文件：data/processed/USDCUSDT_aggTrades_recent_6months.csv
- 时间跨度：2024年11月至2025年5月（7个月，305,280分钟）
- 数据类型：逐笔交易聚合为分钟级流动性特征

💰 价格基准：best_bid（买一价）
- 当前价格：使用best_bid作为close价格基准
- 预测目标：未来5/10/15分钟后的bid价格变化
- 计算公式：(future_bid - current_bid) / current_bid

📈 预测任务：多时间尺度三分类
- 5分钟预测：5分钟后的price时刻变化方向
- 10分钟预测：10分钟后的price时刻变化方向  
- 15分钟预测：15分钟后的price时刻变化方向
- 分类：0=上涨(>+0.01%), 1=下跌(<-0.01%), 2=持平(±0.01%内)

🔧 特征结构：25维特征（更新版）
- 市场深度特征(5个)：bid_depth_1, ask_depth_1, depth_imbalance, total_depth, depth_ratio
- 价格特征(2个)：best_bid, best_ask（移除spread相关）
- 成交量特征(4个)：买卖成交量、成交量比例
- 流动性特征(4个)：交易笔数、大单相关指标
- 市场冲击特征(2个)：价格波动、成交量冲击
- 时间特征(8个)：小时、星期、月日的周期性编码

⚖️ 类别分布特点：极度不平衡
- 持平类占95%+（符合稳定币特性）
- 上涨/下跌类各占2-3%
- 需要专门的类别平衡策略

🎲 模型挑战：
- 捕捉稳定币的微小价格变化（0.01%级别）
- 处理极度不平衡的类别分布
- 基于流动性特征而非传统技术指标进行预测

最后更新：2025-06-02
═══════════════════════════════════════════════════════════════════
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ExperimentConfig:
    """实验配置类 - 所有超参数集中管理"""
    
    # ========== 实验基础设置 ==========
    experiment_name: str = "kline_predictor_v1"
    random_seed: int = 42
    device: str = "cuda"  # 自动检测GPU，fallback到CPU
    
    # ========== 数据相关 ==========
    # CSV数据文件路径（使用新下载的aggTrades数据）
    data_file: str = "data/processed/USDCUSDT_aggTrades_recent_6months.csv"
    
    # 数据列名映射（根据aggTrades数据调整）
    csv_columns: Dict[str, str] = field(default_factory=lambda: {
        'timestamp': 'timestamp',
        'best_bid': 'best_bid',
        'best_ask': 'best_ask', 
        'spread': 'spread',
        'spread_bps': 'spread_bps',
        'total_volume': 'total_volume',
        'buy_volume': 'buy_volume',
        'sell_volume': 'sell_volume',
        'buy_ratio': 'buy_ratio',
        'trade_count': 'trade_count',
        'avg_trade_size': 'avg_trade_size',
        'large_trade_count': 'large_trade_count',
        'large_trade_ratio': 'large_trade_ratio',
        'price_range': 'price_range',
        'volume_impact': 'volume_impact'
    })
    
    # 时间窗口设置
    sequence_length: int = 30  # 输入30分钟历史数据
    prediction_horizons: List[int] = field(default_factory=lambda: [5, 10, 15])  # 预测5、10、15分钟
    
    # 数据分割比例
    train_ratio: float = 0.8   # 80%训练
    val_ratio: float = 0.1     # 10%验证
    test_ratio: float = 0.1    # 10%测试
    
    # 分类阈值（价格变化百分比）
    price_change_threshold: float = 0.0001  # 0.01%作为涨跌分界
    
    # ========== 特征工程 ==========
    # aggTrades核心特征（移除spread，添加市场深度特征）
    aggtrades_features: List[str] = field(default_factory=lambda: [
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
    ])
    
    # 特征总维度（17个aggTrades特征 + 8个时间特征）
    feature_dim: int = 25
    
    # ========== 模型架构 ==========
    model_type: str = "transformer"  # 简化为单一transformer模型
    
    # 分类模式配置
    use_binary_classification: bool = True  # 稳定性检测二分类
    num_classes: int = 2  # 二分类：稳定(0) vs 偏离(1)
    
    # Transformer配置
    transformer_config: Dict[str, Any] = field(default_factory=lambda: {
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'ff_dim': 1024,
        'dropout': 0.1,
        'num_classes': 2  # 稳定性检测二分类
    })
    
    # ========== 训练超参数 ==========
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 100
    
    # 早停设置
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # 学习率调度
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau"
    lr_scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'T_max': 100,  # cosine退火的周期
        'eta_min': 1e-6  # 最小学习率
    })
    
    # ========== 数据增强 ==========
    use_data_augmentation: bool = True
    augmentation_config: Dict[str, Any] = field(default_factory=lambda: {
        'noise_level': 0.01,      # 高斯噪声标准差
        'dropout_rate': 0.05,     # 随机置零比例
        'time_shift_range': 3     # 时间偏移范围
    })
    
    # ========== 损失函数权重 ==========
    # 类别权重 (用于处理不平衡数据) - 稳定性检测二分类
    class_weights: List[float] = field(default_factory=lambda: [
        1.0,   # 稳定类权重（多数类）
        8.0,   # 偏离类权重（少数类，适度提高）
    ])
    
    # 多任务损失权重
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        '5min': 0.5,   # 5分钟预测权重
        '10min': 0.3,  # 10分钟预测权重
        '15min': 0.2   # 15分钟预测权重
    })
    
    # ========== 输出设置 ==========
    # 保存路径
    output_dir: str = "experiments/outputs"
    model_save_dir: str = "experiments/outputs/models"
    log_dir: str = "experiments/outputs/logs"
    plot_dir: str = "experiments/outputs/plots"
    
    # 保存频率
    save_every_n_epochs: int = 10
    log_every_n_steps: int = 100
    
    # 是否绘制详细图表
    plot_training_curves: bool = True
    plot_confusion_matrix: bool = True
    plot_feature_importance: bool = True
    
    # ========== 评估设置 ==========
    # 简化评估策略（稳定性检测）
    evaluation_strategy: str = "traditional"  # 使用传统准确率评估
    
    # 核心评估指标（稳定性检测）
    primary_metric: str = "accuracy"  # 主指标：分类准确率
    secondary_metric: str = "precision"  # 次要指标：精确率
    
    # 简化评估指标
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        'accuracy',           # 分类准确率
        'precision',          # 精确率
        'recall',             # 召回率
        'f1_score',           # F1分数
        'confusion_matrix'    # 混淆矩阵
    ])
    
    # 早停配置（基于准确率）
    early_stopping_metric: str = "accuracy"  # 基于准确率早停
    early_stopping_mode: str = "maximize"    # 最大化模式
    early_stopping_patience: int = 10        # 早停耐心
    early_stopping_min_delta: float = 0.001  # 最小改进阈值
    
    # 模型保存策略（简化）
    save_multiple_models: bool = False
    model_save_criteria: Dict[str, str] = field(default_factory=lambda: {
        'best_model.pth': 'accuracy'  # 最佳准确率模型
    })

    # ========== 损失函数配置 ==========
    # 使用标准二分类损失函数
    use_trading_aware_loss: bool = False  # 使用简单二分类交叉熵损失
    use_binary_classification: bool = True  # 确保二分类模式
    
    def __post_init__(self):
        """初始化后的验证和设置"""
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # 验证配置
        assert sum([self.train_ratio, self.val_ratio, self.test_ratio]) == 1.0
        
        # 二分类模式验证
        assert self.use_binary_classification, "当前只支持稳定性检测二分类模式"
        assert len(self.class_weights) == 2, "二分类模式需要2个类别权重"
        
        # 更新配置以匹配二分类
        self.num_classes = 2
        self.transformer_config['num_classes'] = 2
        
        assert sum(self.task_weights.values()) == 1.0
        
    def get_model_config(self) -> Dict[str, Any]:
        """获取当前模型的配置"""
        if self.model_type == "transformer":
            return self.transformer_config
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")
    
    def summary(self) -> str:
        """配置摘要"""
        return f"""
🧪 实验配置摘要:
实验名称: {self.experiment_name}
模型类型: {self.model_type}
序列长度: {self.sequence_length}分钟
预测时间: {self.prediction_horizons}分钟
批次大小: {self.batch_size}
学习率: {self.learning_rate}
最大轮次: {self.max_epochs}
特征维度: {self.feature_dim}
数据文件: {self.data_file}
输出目录: {self.output_dir}
        """

# 创建默认配置实例
config = ExperimentConfig()
