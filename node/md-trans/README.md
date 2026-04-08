# md-trans

Markdown 双语翻译 CLI - 将英文 Markdown 文档翻译成中文，输出沉浸式的中英文对照格式。

## 特性

- **两阶段翻译**：预分析建立术语表 + 串行翻译保证一致性
- **基于 AST 处理**：使用 unified/remark 生态精确解析 Markdown 结构
- **术语一致性**：缓存翻译结果，确保同一术语始终翻译一致
- **智能跳过**：自动跳过代码块、行内代码和 HTML
- **双语输出**：每个节点英文在上，中文在下
- **多 LLM 支持**：支持 Anthropic 和 OpenAI 兼容接口

## 安装

```bash
pnpm install
```

## 使用方法

```bash
# 基本用法
md-trans input.md

# 指定输出文件
md-trans input.md -o output.md

# Debug 模式（显示每次 LLM 请求和应答）
md-trans input.md --debug

# 跳过预分析（更快，一致性略差）
md-trans input.md --skip-preanalysis

# 测试 LLM 连接
md-trans --test-llm

# 使用 OpenAI 模式（默认是 Anthropic）
md-trans input.md --llm-mode openai

# 自定义超时（秒，默认 120）
md-trans input.md --timeout 300
```

## 环境变量配置

在 `.env` 文件中配置：

```bash
# Anthropic 模式（默认）
ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
ANTHROPIC_API_KEY=your-api-key
ANTHROPIC_MODEL=MiniMax-M2.7

# OpenAI 模式
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4
```

或通过命令行/环境变量：

```bash
export ANTHROPIC_API_KEY=sk-your-api-key
export ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
export ANTHROPIC_MODEL=MiniMax-M2.7
```

## 输出格式

工具输出双语 Markdown，每个节点先英文后中文：

```markdown
## English Heading

## 中文标题

Some paragraph text in English.

一些中文段落文本。

- English list item

- 中文列表项

```javascript
const x = 1;  // 代码块不翻译
```
```

## 项目结构

- `src/parser/` - Markdown 解析（unified/remark）
- `src/translator/` - LLM API 客户端、预分析器、翻译引擎
- `src/processor/` - 分块、缓存、双语渲染
- `src/config/` - 提示词模板

## 测试

```bash
pnpm test
```

## License

MIT
