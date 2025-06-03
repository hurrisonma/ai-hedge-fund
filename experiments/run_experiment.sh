#!/bin/bash

# 🧪 深度学习实验运行脚本
# 独立实验程序，不影响现有工程

echo "🧪 准备启动深度学习实验..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装Python3"
    exit 1
fi

# 进入实验目录
cd "$(dirname "$0")"

# 检查依赖是否安装
echo "📦 检查依赖包..."
python3 -c "import torch, pandas, numpy, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  缺少依赖包，正在安装..."
    pip3 install torch pandas numpy scikit-learn matplotlib seaborn tqdm
fi

# 创建必要目录
echo "📁 创建输出目录..."
mkdir -p data/raw data/processed data/features
mkdir -p outputs/models outputs/logs outputs/plots

# 设置权限
chmod +x main.py

echo ""
echo "🚀 启动实验..."
echo "=================================================="

# 运行实验（使用默认配置）
python3 main.py "$@"

echo ""
echo "=================================================="
echo "✅ 实验完成！"
echo ""
echo "📂 输出文件位置："
echo "  - 模型文件: outputs/models/"
echo "  - 训练日志: outputs/logs/"
echo "  - 图表文件: outputs/plots/"
echo ""
echo "📊 查看训练日志:"
echo "  tensorboard --logdir=outputs/logs"
echo "" 