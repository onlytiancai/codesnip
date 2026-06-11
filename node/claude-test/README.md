# 沉浸式 Markdown 翻译工具

一个使用 OpenAI 兼容 API 的 Markdown 文档翻译工具，保留原有格式，实现中英文对照翻译。

## 功能特点

- 🎯 **智能分析**：第一步通读全文，制定翻译策略
- 📝 **格式保留**：保持 Markdown 原有的标题、列表、引用等格式
- 💬 **中英对照**：英文在上，中文在下，段落间有空行
- 🚫 **代码保护**：自动跳过代码块和代码片段
- 🌊 **流式翻译**：实时显示翻译进度，支持长时间文档
- 🔧 **灵活配置**：支持任何 OpenAI 兼容的 API 服务

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

1. 复制 `.env.example` 为 `.env`：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，设置你的 API 配置：

```env
# OpenAI 官方 API
OPENAI_API_KEY=sk-your-openai-key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# 或使用其他兼容服务（如 OpenRouter）
# OPENAI_API_BASE=https://openrouter.ai/api/v1
# OPENAI_MODEL=openai/gpt-3.5-turbo
```

## 使用方法

运行脚本并输入要翻译的 Markdown 文件路径：

```bash
python markdown_translator.py
```

程序会：
1. 首先分析文档风格，制定翻译策略
2. 然后逐个节点翻译（段落、标题、列表等）
3. 生成中英文对照的翻译文件

## 输出格式

原文格式：
```markdown
# 标题

段落内容，包含一些 `代码` 和 **强调**。

- 列表项 1
- 列表项 2

> 引用内容
```

翻译后格式：
```markdown
# 标题

段落内容，包含一些 `代码` 和 **强调**。

段落内容，包含一些 `代码` 和 **强调**。

- 列表项 1
- 列表项 1

- 列表项 2
- 列表项 2

> 引用内容
> 引用内容
```

## 支持的 API 服务

- OpenAI 官方 API
- OpenRouter
- LocalAI
- Ollama
- 任何 OpenAI 兼容的自部署服务

## 注意事项

- 代码块和内联代码不会翻译
- 翻译会保持原文的语气和专业性
- 每段翻译后都有一个空行分隔
- 列表项是一行英文对应一行中文