# AI Agent Demo

基于 Anthropic SDK 的 CLI AI Agent，支持 MiniMax API 兼容调用，具备多轮对话和工具调用能力。

## 项目结构

```
scripts/agent-demo/
├── package.json       # 依赖和脚本
├── tsconfig.json      # TypeScript 配置
├── .env.example       # 环境变量模板
├── .env               # 环境变量
├── memory.md          # 持久化全局记忆（用户可修改）
├── memory-system.md   # 内置系统提示词（用户不可修改）
├── memory/            # 时间戳记忆目录
│   └── YYYY-MM-DD.md  # 每日记忆文件
└── src/
    ├── index.ts       # 主入口（CLI、消息流、状态管理）
    ├── tools.ts       # 工具定义与执行
    └── utils.ts       # 工具函数（格式化、错误处理）
```

## 核心能力

### 可用工具

| 工具 | 功能 | 示例 |
|------|------|------|
| `read` | 读取文件内容（路径安全验证） | `read package.json` |
| `write` | 创建/更新文件（自动创建目录） | `write({ path: "dir/file.txt", content: "..." })` |
| `ls` | 列出目录内容 | `ls /Users/huhao/projects` |
| `mkdir` | 创建目录（递归） | `mkdir /tmp/test` |
| `web_fetch` | 获取网页内容并提取文本（限制 5000 字符） | `web_fetch https://example.com` |
| `save_today_memory` | 保存重要信息到今日记忆文件 | `save_today_memory({ content: "用户喜欢用中文交流" })` |
| `save_global_memory` | 保存永久信息到全局记忆 | `save_global_memory({ content: "用户是蛙蛙的主人" })` |

### 功能特性

- **真正的迭代 Tool Loop**：支持多次工具调用循环（最多 10 轮），非固定 2 轮
- **多轮对话**：完整的对话历史支持（最大 100 条消息）
- **并行工具调用**：多个工具同时执行
- **持久化记忆系统**：
  - `memory-system.md` - 内置系统提示词（用户不可修改）
  - `memory.md` - 用户可修改的全局记忆
  - `memory/YYYY-MM-DD.md` - 每日时间戳记忆（启动时加载今天和昨天）
- **记忆保存工具**：`save_today_memory` 和 `save_global_memory` 供 Agent 主动保存重要信息
- **Thinking 块**：显示模型的推理过程
- **流式响应**：实时输出，带状态指示
- **自动重试**：API 失败时自动重试（429、5xx、网络错误、timeout）
- **斜杠命令**：
  - `/new` - 清空历史，开启新对话
  - `/exit` - 退出程序
- **非交互模式**：命令行直接传入 prompt，单次执行后退出
- **指标收集**：统计请求数、token 用量、工具调用次数、错误数、执行时长

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ANTHROPIC_BASE_URL` | API 端点 | `https://api.anthropic.com` |
| `ANTHROPIC_API_KEY` | 你的 API 密钥 | （必填） |
| `ANTHROPIC_MODEL` | 使用的模型 | `claude-3-5-sonnet-20241022` |
| `ANTHROPIC_TIMEOUT_MS` | 请求超时时间（毫秒） | `120000` (120s) |

## 快速开始

```bash
cd scripts/agent-demo
pnpm install
pnpm dev
```

## 使用方式

### 交互模式（默认）

启动后进入 REPL，可多次输入：

```bash
pnpm dev
> 你好，你是谁？
> 读取当前目录的 package.json
> /exit
```

### 非交互模式

命令行直接传入 prompt，执行完成后自动退出：

```bash
pnpm dev "你好，你是谁？"
pnpm dev "列出当前目录的文件"
pnpm dev "抓取 https://example.com 的内容并总结"
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

### 2. 记忆系统使用

Agent 可以主动保存重要信息：

```
> 用户告诉我他叫蛙蛙，请记住这个信息
> 用户喜欢用中文交流，请把这个偏好也记住
```

Agent 会使用 `save_global_memory` 和 `save_today_memory` 工具保存信息。

### 3. 文件操作

```
> 创建一个叫 hello.txt 的文件，内容是 "Hello World"
> 读取当前目录
> 列出 /tmp 下的文件
```

### 4. 网页获取和摘要

```
> 获取 https://news.ycombinator.com 并总结热门内容
> 抓取 https://blog.ihuhao.com/2026/03/19/spx-iron-condor/ 的内容
```

### 5. 复杂多步骤任务

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

### 6. 多轮调试对话

```
> 读取我的 package.json
> 测试脚本失败了，帮我排查一下？
> 请修复这个错误
```

### 7. 非交互模式执行脚本任务

```bash
# 单次执行并退出，适合脚本集成
pnpm dev "在当前目录创建 output.txt，内容是 'Done'"

# 配合其他命令使用
pnpm dev "读取 memory.md 的内容" && cat memory.md
```

## 安全特性

- **路径验证**：所有文件操作限制在允许目录下，防止路径遍历攻击
- **请求超时**：可配置的请求超时，避免无限等待
- **资源清理**：正确关闭 readline 句柄

## 指标输出

执行完成后会输出统计信息：

```
[Metrics] Requests: 2, Tokens: 1523, ToolCalls: 3, Errors: 0, Duration: 12.34s
```

## 快捷键

- `Ctrl+C` - 退出（干净退出，无需更新记忆）

## 测试 Prompt

- 基础：`你好，你是谁？我是谁？`
- 文件：`在当前目录创建 test.txt，内容是 "hello"`
- 网页：`抓取 https://blog.ihuhao.com/ 并总结主要内容`
- 复杂任务：`在当前目录下创建一个 test 目录，抓取 https://blog.ihuhao.com/2026/03/19/spx-iron-condor/ 页面的内容后做摘要并保存在这个目录`
