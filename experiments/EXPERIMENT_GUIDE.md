# 🧪 深度学习实验使用指南

## ⚠️ 重要说明

**这是一个完全独立的实验性程序，对现有工程没有任何影响！**

- 🔒 所有代码都在 `experiments/` 目录下
- 📦 使用独立的依赖环境
- 🧪 纯实验性质，用于验证深度学习方案

## 🎯 实验目标

基于1年分钟级K线数据，使用Transformer深度学习模型预测未来5/10/15分钟的价格趋势（上涨/下跌/持平）。

## 🚀 快速开始

### 方法1：一键运行（推荐）

```bash
cd experiments/
./run_experiment.sh
```

### 方法2：手动运行

```bash
cd experiments/

# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行实验
python main.py
```

### 方法3：自定义参数

```bash
cd experiments/

# 使用自己的数据文件
python main.py --data /path/to/your/kline_data.csv

# 调整训练参数
python main.py --epochs 100 --batch-size 128 --lr 0.0001

# 组合使用
python main.py \
  --data data/raw/my_data.csv \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001
```

## 📊 数据格式要求

### CSV文件格式

你的K线数据CSV文件需要包含以下列：

```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.0856,1.0859,1.0854,1.0857,156789
2024-01-01 00:01:00,1.0857,1.0860,1.0855,1.0858,134567
2024-01-01 00:02:00,1.0858,1.0861,1.0856,1.0859,123456
...
```

### 数据要求

- ✅ **时间跨度**：建议至少6个月的数据（实验会自动生成1年示例数据）
- ✅ **时间间隔**：1分钟级别的K线数据
- ✅ **数据完整性**：尽量避免缺失值和异常值
- ✅ **文件大小**：支持大文件，内存会自动优化

### 如果没有数据

程序会自动创建1年的模拟K线数据进行实验，数据特征：
- 基础价格：1.0856（EUR/USD类似）
- 随机游走模式，符合金融时序特征
- 包含完整的OHLCV数据

## 🏗️ 程序架构

```
experiments/
├── main.py              # 🚀 主程序入口
├── training/            # 训练相关
│   ├── config.py       # ⚙️ 配置文件（重要！）
│   └── data_loader.py  # 📊 数据加载器
├── models/             # 模型定义
│   └── transformer.py  # 🧠 Transformer模型
├── utils/              # 工具函数
│   └── feature_engineer.py  # 🔧 特征工程
├── data/               # 数据目录
│   ├── raw/           # 原始CSV数据
│   ├── processed/     # 预处理后数据
│   └── features/      # 特征工程结果
└── outputs/            # 输出结果
    ├── models/        # 💾 训练好的模型
    ├── logs/          # 📈 训练日志
    └── plots/         # 📊 图表文件
```

## ⚙️ 配置说明

核心配置在 `training/config.py` 中，主要参数：

### 数据配置
```python
sequence_length: 30        # 输入30分钟历史数据
prediction_horizons: [5, 10, 15]  # 预测5、10、15分钟
price_change_threshold: 0.001      # 0.1%作为涨跌分界
```

### 模型配置
```python
embed_dim: 256            # 嵌入维度
num_heads: 8              # 注意力头数
num_layers: 6             # Transformer层数
dropout: 0.1              # Dropout比例
```

### 训练配置
```python
batch_size: 64            # 批次大小
learning_rate: 0.001      # 学习率
max_epochs: 100           # 最大训练轮数
early_stopping_patience: 10  # 早停耐心值
```

## 📈 运行过程

程序运行分为5个阶段：

### 1. 📊 数据准备阶段
- 加载CSV数据
- 特征工程（38维特征）
- 创建预测标签
- 数据分割（训练/验证/测试）

### 2. 🏗️ 模型构建阶段
- 构建Transformer模型
- 设置优化器和损失函数
- 初始化TensorBoard日志

### 3. 🚀 模型训练阶段
- 自动训练循环
- 实时显示损失和准确率
- 自动保存最佳模型
- 早停机制防止过拟合

### 4. 📊 模型评估阶段
- 在测试集上评估
- 计算各种指标（准确率、精确率、F1等）
- 生成混淆矩阵

### 5. 📈 结果可视化
- 训练曲线图
- 模型性能图表
- TensorBoard日志

## 📊 结果输出

### 控制台输出示例

```
🧪 实验配置摘要:
实验名称: kline_predictor_v1
模型类型: transformer
序列长度: 30分钟
预测时间: [5, 10, 15]分钟
批次大小: 64
学习率: 0.001

📊 数据准备阶段
✅ 成功加载 525600 行数据
🔧 开始特征工程...
✅ 特征工程完成，总特征数: 38

🏗️ 模型构建阶段
📊 模型参数统计:
  总参数: 2,847,619
  可训练参数: 2,847,619
  模型大小: ~10.9MB

🚀 模型训练阶段
📈 Epoch 1/100
训练损失: 1.0856 | 训练准确率: 0.456
验证损失: 1.0234 | 验证准确率: 0.523
各时间尺度准确率:
  5min: 0.534
  10min: 0.518
  15min: 0.517
💾 保存最佳模型

📊 模型评估阶段
5min 预测结果:
  准确率: 0.678
  精确率: 0.682
  召回率: 0.678
  F1分数: 0.679
```

### 输出文件

- **模型文件**: `outputs/models/best_model.pth`
- **训练日志**: `outputs/logs/` (TensorBoard格式)
- **图表文件**: `outputs/plots/training_curves.png`
- **预处理器**: `outputs/preprocessor.pkl`

## 🔍 查看结果

### TensorBoard可视化

```bash
cd experiments/
tensorboard --logdir=outputs/logs
# 然后打开浏览器访问 http://localhost:6006
```

### 使用训练好的模型

```python
import torch
from models.transformer import MarketTransformer

# 加载模型配置和权重
config = {...}  # 模型配置
model = MarketTransformer(config)
model.load_state_dict(torch.load('outputs/models/best_model.pth'))

# 进行预测
predictions = model.predict_proba(new_data)
```

## 🛠️ 常见问题

### Q: 内存不足怎么办？
A: 减小批次大小：`python main.py --batch-size 32`

### Q: 训练太慢怎么办？  
A: 减少训练轮数：`python main.py --epochs 20`

### Q: 想使用GPU训练？
A: 程序会自动检测并使用GPU，确保安装了CUDA版本的PyTorch

### Q: 如何调整模型复杂度？
A: 修改 `training/config.py` 中的 `transformer_config` 参数

### Q: 预测准确率不高怎么办？
A: 
- 增加训练数据量
- 调整 `price_change_threshold` 参数
- 尝试不同的学习率
- 增加模型复杂度

## 🧪 实验变体

### 实验1：不同阈值测试
```bash
# 测试更敏感的阈值（0.05%）
python main.py --threshold 0.0005

# 测试更保守的阈值（0.2%）  
python main.py --threshold 0.002
```

### 实验2：不同时间窗口
修改 `config.py` 中的：
```python
sequence_length: 60  # 使用60分钟历史数据
prediction_horizons: [10, 20, 30]  # 预测更长时间
```

### 实验3：模型对比
```python
# 在config.py中切换模型类型
model_type: "cnn_lstm"  # 使用CNN+LSTM模型
```

## 📝 注意事项

1. **实验性质**：这是学术研究程序，不构成投资建议
2. **数据隐私**：程序只在本地运行，不会上传任何数据
3. **计算资源**：建议至少8GB内存，训练时间取决于数据量和硬件
4. **结果解释**：准确率仅供参考，实际市场应用需要更多验证

## 📞 技术支持

如果遇到问题，可以：
1. 查看控制台错误信息
2. 检查 `outputs/logs/` 中的详细日志
3. 确认数据格式是否正确
4. 验证Python依赖是否完整安装

---

**�� 开始你的深度学习实验之旅吧！** 