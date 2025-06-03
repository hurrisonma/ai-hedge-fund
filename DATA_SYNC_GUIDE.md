# 📊 数据同步指南

## 🚫 不同步的文件类型

本项目已配置`.gitignore`来排除以下类型的文件，这些文件**不会**被Git同步：

### 📈 数据文件
- `*.csv` - CSV数据文件（包括所有交易数据）
- `*.json` - JSON数据文件
- `*.pickle`, `*.pkl` - Python序列化文件
- `*.h5`, `*.hdf5` - HDF5数据文件
- `*.parquet`, `*.feather` - 列式存储文件
- `*.xlsx`, `*.xls` - Excel文件
- `*.sqlite`, `*.db` - 数据库文件

### 🧠 模型文件
- `*.pth`, `*.pt` - PyTorch模型文件
- `*.bin`, `*.safetensors` - 二进制模型文件
- `*.ckpt` - 检查点文件
- `*.model` - 通用模型文件

### 📁 数据目录
- `experiments/data/raw/` - 原始数据目录
- `experiments/data/processed/` - 处理后数据目录
- `experiments/outputs/` - 实验输出目录
- `logs/` - 日志目录

### 🖼️ 媒体文件
- `*.png`, `*.jpg`, `*.pdf` - 图片和文档文件

## ✅ 同步的文件

以下文件**会**被Git同步：

### 💻 代码文件
- `*.py` - Python源代码
- `*.js`, `*.ts` - JavaScript/TypeScript代码
- `*.html`, `*.css` - 前端代码

### ⚙️ 配置文件
- `*.yml`, `*.yaml` - 配置文件
- `*.toml` - 项目配置（如pyproject.toml）
- `*.json` - 配置相关的JSON文件（非数据）
- `requirements.txt` - 依赖文件

### 📝 文档文件
- `*.md` - Markdown文档
- `README.md` - 项目说明

## 🔄 数据获取方式

由于数据文件不同步到Git，团队成员需要通过以下方式获取数据：

### 1. 运行数据下载脚本
```bash
# 进入实验目录
cd experiments

# 运行币安数据下载脚本
python aggtrades_downloader.py

# 或使用shell脚本
./download_binance_data.sh
```

### 2. 手动下载数据
如果自动下载失败，可以：
1. 从币安API手动获取数据
2. 从其他团队成员处获取数据文件
3. 使用测试数据进行开发

### 3. 数据目录结构
下载后的数据应该放在：
```
experiments/data/
├── raw/              # 原始数据
│   ├── aggtrades/   # aggTrades数据
│   └── binance/     # 币安API数据
└── processed/       # 处理后数据
    └── *.csv        # 处理好的CSV文件
```

## ⚠️ 注意事项

1. **不要手动添加数据文件到Git**
   ```bash
   # ❌ 错误做法
   git add experiments/data/processed/*.csv
   
   # ✅ 正确做法 - 这些文件会被自动忽略
   git status  # 不会显示这些文件
   ```

2. **检查文件大小**
   - 数据文件通常很大（几十MB到几GB）
   - Git不适合存储大文件
   - 如有必要，考虑使用Git LFS

3. **本地开发**
   - 确保本地有足够的存储空间
   - 定期清理不需要的数据文件
   - 使用`.gitignore`测试确认文件不会被追踪

## 🔍 验证配置

检查`.gitignore`是否生效：
```bash
# 查看Git状态，数据文件不应该出现
git status

# 检查特定文件是否被忽略
git check-ignore experiments/data/processed/USDCUSDT_aggTrades_recent_6months.csv

# 查看所有被忽略的文件
git status --ignored
```

## 📞 问题反馈

如果遇到数据同步相关问题，请：
1. 检查`.gitignore`配置
2. 确认数据文件路径
3. 联系项目维护者 