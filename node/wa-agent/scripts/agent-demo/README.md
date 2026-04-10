# AI Agent Demo

基于 Anthropic SDK 的 CLI AI Agent，支持 MiniMax API 兼容调用，具备多轮对话和工具调用能力。

## 项目结构

```
scripts/agent-demo/
├── package.json       # 依赖和脚本
├── tsconfig.json      # TypeScript 配置
├── .env.example       # 环境变量模板
├── .env               # 环境变量
├── memory.md          # 持久化记忆（作为 system prompt）
└── src/
    └── index.ts       # 单文件实现（客户端 + 工具 + 主循环）
```

## 核心能力

### 可用工具

| 工具 | 功能 | 示例 |
|------|------|------|
| `read` | 读取文件内容 | `read package.json` |
| `write` | 创建/更新文件（自动创建目录） | `write({ path: "dir/file.txt", content: "..." })` |
| `ls` | 列出目录内容 | `ls /Users/huhao/projects` |
| `mkdir` | 创建目录（递归） | `mkdir /tmp/test` |
| `web_fetch` | 获取网页内容并提取文本 | `web_fetch https://example.com` |

### 功能特性

- **多轮对话**：完整的对话历史支持
- **并行工具调用**：多个工具同时执行
- **持久化记忆**：`memory.md` 作为 system prompt 提供长期上下文
- **Thinking 块**：显示模型的推理过程
- **流式响应**：实时输出，带状态指示
- **自动重试**：API 失败时自动重试（429、5xx、网络错误）
- **斜杠命令**：
  - `/new` - 清空历史，开启新对话
  - `/exit` - 退出程序

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ANTHROPIC_BASE_URL` | API 端点 | `https://api.minimaxi.com/anthropic` |
| `ANTHROPIC_API_KEY` | 你的 API 密钥 | （必填） |
| `ANTHROPIC_MODEL` | 使用的模型 | `MiniMax-M2.7` |

## 快速开始

```bash
cd scripts/agent-demo
pnpm install
pnpm dev
```

## 使用示例

### 1. 根据记忆回答身份问题

如果 `memory.md` 包含：
```markdown
- 你的名字叫毛毛
- 你的主人叫蛙蛙
- 请用中文回答
```

Agent 会知道自己的身份并用中文回复。

### 2. 文件操作

```
> 创建一个叫 hello.txt 的文件，内容是 "Hello World"
> 读取当前目录
> 列出 /tmp 下的文件
```

### 3. 网页获取和摘要

```
> 获取 https://news.ycombinator.com 并总结热门内容
> 抓取 https://blog.ihuhao.com/2026/03/19/spx-iron-condor/ 的内容
```

### 4. 复杂多步骤任务

```
> 在当前目录下创建 test 文件夹，抓取
  https://blog.ihuhao.com/2026/03/19/spx-iron-condor/ 的内容，
  摘要后保存到 test/summary.txt
```

Agent 会自动完成：
1. 创建 `test` 目录
2. 获取网页内容
3. 总结内容
4. 保存到 `test/summary.txt`

### 5. 多轮调试对话

```
> 读取我的 package.json
> 测试脚本失败了，帮我排查一下？
> 请修复这个错误
```

## 快捷键

- `Ctrl+C` - 退出（干净退出，无需更新记忆）

## 测试 Prompt

- 基础：`你好，你是谁？我是谁？`
- 文件：`在当前目录创建 test.txt，内容是 "hello"`
- 网页：`抓取 https://blog.ihuhao.com/ 并总结主要内容`
- 复杂任务：`在当前目录下创建一个 test 目录，抓取 https://blog.ihuhao.com/2026/03/19/spx-iron-condor/ 页面的内容后做摘要并保存在这个目录`
