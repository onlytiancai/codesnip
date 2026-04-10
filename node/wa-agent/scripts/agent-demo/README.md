# Agent Demo

基于官方 Anthropic SDK 的 CLI AI Agent 演示，支持 MiniMax API 兼容调用。

## 项目结构

```
scripts/agent-demo/
├── package.json       # 依赖和脚本
├── tsconfig.json      # TypeScript 配置
├── .env.example       # 环境变量模板
├── .env               # 环境变量（从 .env.example 复制）
├── memory.md          # 持久化记忆文件（作为 system prompt）
├── src/
│   ├── index.ts       # 主入口，CLI 循环
│   ├── tools.ts       # 工具定义（read, write, ls, mkdir, web_fetch）
│   └── client.ts      # AI 客户端配置
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ANTHROPIC_BASE_URL` | API 端点 | `https://api.minimaxi.com/anthropic` |
| `ANTHROPIC_API_KEY` | 你的 API 密钥 | （必填） |
| `ANTHROPIC_MODEL` | 使用的模型 | `MiniMax-M2.7` |

## 功能特性

- **多轮对话**：支持对话历史
- **内置工具**：`read`、`write`、`ls`、`mkdir`、`web_fetch`
- **记忆持久化**：`memory.md` 文件作为 system prompt
- **实时状态显示**：流式响应时显示请求状态
- **简洁 CLI 界面**：简单的命令行交互
- **自动重试**：API 请求失败时自动重试（最多 3 次，延迟 1s/2s/4s）
- **Thinking 支持**：完整支持 MiniMax 模型的 thinking blocks

## 使用方法

```bash
cd scripts/agent-demo
pnpm install
pnpm dev
```

### 运行 Agent

1. 启动时 Agent 会加载 `memory.md` 作为 system prompt
2. 输入消息后按 Enter 发送
3. Agent 可以调用工具来回答问题
4. 按 Ctrl+C 退出

### 退出时

按 Ctrl+C 后：
- 会显示对话轮数
- 提示是否更新 `memory.md`
- 如选择更新，新内容会被保存

## 常见问题

### 记忆文件加载失败

确保 `memory.md` 存在于项目根目录。如果不存在，将使用默认记忆。

### API 请求失败

程序内置自动重试机制，对以下错误自动重试：
- 429 Rate Limit
- 500/502/503 Server Error
- Network Error

重试延迟：1秒 → 2秒 → 4秒（最多 3 次）

## 测试 prompt

- 在当前目录下创建一个test目录,抓取https://blog.ihuhao.com/2026/03/19/spx-iron-condor/页面的内容后做摘要并保存在这个目录