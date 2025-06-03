# 🧪 深度学习实验程序

## ⚠️ 重要说明

**这是一个独立的实验性程序，对现有工程没有任何影响！**

- 📁 所有实验代码都在 `experiments/` 目录下
- 🔒 完全独立运行，不会修改现有的 `src/` 目录
- 🧪 纯实验性质，用于验证深度学习方案的可行性
- 📊 基于K线数据的时序预测模型实验

## 实验目标

基于1年分钟级K线数据，使用深度学习模型预测未来5/10/15分钟的价格趋势（上涨/下跌/持平）。

## 目录结构

```
experiments/
├── README.md              # 本文档
├── requirements.txt       # 实验专用依赖
├── data/                  # 数据目录
│   ├── raw/              # 原始CSV数据
│   ├── processed/        # 预处理后数据
│   └── features/         # 特征工程结果
├── models/               # 模型定义
│   ├── transformer.py    # Transformer模型
│   ├── cnn_lstm.py      # CNN+LSTM混合模型
│   └── base_model.py    # 基础模型类
├── training/             # 训练相关
│   ├── trainer.py       # 训练器
│   ├── data_loader.py   # 数据加载器
│   └── config.py        # 配置文件
├── evaluation/           # 评估工具
│   ├── metrics.py       # 评估指标
│   └── visualizer.py    # 可视化工具
├── utils/               # 工具函数
│   ├── feature_engineer.py  # 特征工程
│   ├── data_processor.py    # 数据处理
│   └── helpers.py           # 辅助函数
└── notebooks/           # Jupyter实验笔记
    └── experiment_log.ipynb
```

## 运行方式

```bash
# 进入实验目录
cd experiments/

# 安装依赖
pip install -r requirements.txt

# 运行完整实验
python main.py

# 或分步骤运行
python training/data_loader.py   # 数据预处理
python training/trainer.py       # 模型训练
python evaluation/metrics.py     # 结果评估
```

## 实验数据

- 📊 输入：1年分钟级K线CSV数据
- 🔄 处理：自动特征工程、时序分割
- 🎯 输出：多时间尺度概率预测模型

## 免责声明

此程序仅用于技术实验和学术研究，不构成任何投资建议。 