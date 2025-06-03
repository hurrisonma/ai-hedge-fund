#!/bin/bash

# 📊 币安数据下载脚本
# 快速下载真实的USDCUSDT数据用于深度学习训练

echo "📊 币安数据下载器"
echo "=================================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装Python3"
    exit 1
fi

# 进入实验目录
cd "$(dirname "$0")"

# 检查依赖
echo "📦 检查依赖包..."
python3 -c "import requests, pandas, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  缺少依赖包，正在安装..."
    pip3 install requests pandas tqdm
fi

echo ""
echo "🚀 启动数据下载程序..."
echo "=================================================="

# 运行数据下载器
python3 data_downloader.py

echo ""
echo "✅ 数据下载完成！"
echo ""
echo "📂 下载的数据保存在 data/processed/ 目录"
echo "💡 在深度学习程序中修改配置文件使用真实数据" 