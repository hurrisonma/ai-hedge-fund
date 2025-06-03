# 硅基流动(SiliconFlow) LLM 配置说明

## 配置完成状态

✅ **后端配置已完成**
- 已在 `src/llm/models.py` 中添加 `SILICONFLOW` 提供商支持
- 已在 `src/llm/api_models.json` 中添加硅基流动模型配置
- 已在 `app/frontend/src/data/models.ts` 中同步前端模型选择器

✅ **支持的模型**
- `Pro/deepseek-ai/DeepSeek-R1` (显示为 "DeepSeek R1 Pro")
- `Qwen/Qwen3-32B` (显示为 "Qwen3 32B")

## 需要手动完成的配置

### 1. 创建或更新 .env 文件

在项目根目录创建 `.env` 文件，添加以下配置：

```bash
# 复制现有 .env 文件（如果存在）或创建新的
cp .env.example .env

# 编辑 .env 文件，添加硅基流动配置
```

### 2. 在 .env 文件中添加硅基流动配置

```bash
# SiliconFlow API Configuration
SILICONFLOW_API_KEY=sk-btcdqmbsipcyhlketsidewfzokdxzzvjaryudekqljruenzq
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

### 3. 完整的 .env 文件模板

```bash
# AI Hedge Fund Environment Variables

# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=

# Groq API Configuration
GROQ_API_KEY=your-groq-api-key

# Anthropic API Configuration
ANTHROPIC_API_KEY=your-anthropic-api-key

# DeepSeek API Configuration
DEEPSEEK_API_KEY=your-deepseek-api-key

# Google Gemini API Configuration
GOOGLE_API_KEY=your-google-api-key

# SiliconFlow API Configuration
SILICONFLOW_API_KEY=sk-btcdqmbsipcyhlketsidewfzokdxzzvjaryudekqljruenzq
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# Financial Data API Configuration
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key

# Ollama Configuration (for local models)
OLLAMA_HOST=localhost
OLLAMA_BASE_URL=http://localhost:11434
```

## 测试配置

配置完成后，您可以：

1. **命令行测试**：
   ```bash
   poetry run python src/main.py --ticker AAPL
   ```
   在模型选择界面中应该能看到硅基流动的模型选项

2. **Web界面测试**：
   ```bash
   cd app && ./run.sh
   ```
   在前端模型选择器中应该能看到 "DeepSeek R1 Pro" 和 "Qwen3 32B" 选项

## 注意事项

- 硅基流动使用 OpenAI 兼容的 API 格式
- API 密钥已包含在配置中：`sk-btcdqmbsipcyhlketsidewfzokdxzzvjaryudekqljruenzq`
- 默认 API 地址：`https://api.siliconflow.cn/v1`
- 模型名称保持硅基流动的原始格式

## bc错误处理

如果遇到硅基流动相关的bc错误：

**错误原因**：
- API 密钥未正确配置在 .env 文件中
- SILICONFLOW_API_KEY 环境变量未正确读取
- 网络连接问题或 API 地址不正确

**解决方法**：
1. 检查 .env 文件是否存在于项目根目录
2. 确认 .env 文件中包含正确的 SILICONFLOW_API_KEY
3. 重新启动应用程序以重新加载环境变量
4. 检查网络连接和防火墙设置

**手工拷贝修复**：
如果自动配置失败，请手动拷贝上述环境变量配置到您的 .env 文件中。 