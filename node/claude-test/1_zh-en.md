<!--
  Converted from: 1.html
  Generated on: 2026-02-25T02:12:41.668Z
  Clean mode: false
-->

https://x.com/elvissun/status/2025920521871716562?s=61

I don't use Codex `/ˈkoʊdeks/` or Claude `/klɔːd/` Code directly anymore.

我不再直接使用 Codex 或 Claude Code。

I use OpenClaw `/klɔː/` as my orchestration `/ˌɔːrkɪˈstreɪʃn/` layer. My orchestrator, Zoe, spawns `/spɔːnz/` the agents, writes their prompts, picks the right model for each task, monitors progress, and pings me on Telegram when PRs are ready to merge.

我使用 OpenClaw 作为编排层。我的编排器 Zoe 会生成代理、编写它们的提示、为每个任务选择合适的模型、监控进度，并在 PR 准备好合并时通过 Telegram 通知我。

Proof points from the last 4 weeks:

过去 4 周的证据：

- 94 commits in one day. My most productive day - I had 3 client calls and didn't open my editor once. The average is around 50 commits a day.

- 一天提交 94 次。我最高效的一天 - 我进行了 3 次客户通话，一次都没有打开编辑器。平均每天大约提交 50 次。

- 7 PRs in 30 minutes. Idea to production are blazing `/ˈbleɪzɪŋ/` fast because coding and validations are mostly automated.

- 30 分钟内创建 7 个 PR。从想法到生产环境的速度非常快，因为编码和验证大多是自动化的。

- Commits → MRR: I use this for a real B2B SaaS I'm building — bundling it with founder-led sales to deliver most feature requests same-day. Speed converts leads into paying customers.

- 提交 → 月度经常性收入（MRR）：我正在为构建的真实 B2B SaaS 使用它——将其与创始人主导的销售捆绑在一起，在当天交付大部分功能请求。加速潜在客户到付费客户的转换。

[![图像](https://pbs.twimg.com/media/HByXnBmW8AANOl9?format=jpg&name=medium)](/elvissun/article/2025920521871716562/media/2025660629109895168)

before Jan: CC/codex only | after Jan: Openclaw orchestrates CC/codex

1月之前：只使用 CC/codex | 1月之后：Openclaw 编排 CC/codex

My git history looks like I just hired a dev team. In reality `/riˈæləti/` it's just me going from managing claude code, to managing an openclaw agent that manages a fleet of other claude code and codex agents.

我的 git 历史记录看起来就像我刚刚雇佣了一个开发团队。实际上，我只是从管理 Claude Code，转变到管理一个管理着一群其他 Claude Code 和 Codex 代理的 OpenClaw 代理。

Success rate: The system one-shots almost all small to medium `/ˈmiːdiəm/` tasks without any intervention `/ˌɪntərˈvenʃ(ə)n/`.

成功率：该系统几乎一次性完成所有小型到中型任务，无需任何干预。

Cost: ~$100/month for Claude and $90/month for Codex, but you can start with $20.

成本：Claude 约 $100/月，Codex $90/月，但你可以从 $20 开始。

Here's why this works better than using Codex or Claude Code directly:

以下是为什么这比直接使用 Codex 或 Claude Code 更有效的原因：

> Codex and Claude Code have very little context about your business.

> Codex 和 Claude Code 对你的业务背景了解很少。

They see code. They don't see the full picture of your business.

他们看到代码。他们看不到你的业务的全貌。

OpenClaw changes the equation. It acts as the orchestration layer between you and all agents — it holds all my business context (customer data, meeting notes, past decisions, what worked, what failed) inside my Obsidian vault `/vɔːlt/`, and translates historical context into precise `/prɪˈsaɪs/` prompts for each coding agent. The agents stay focused on code. The orchestrator stays at the high strategy `/ˈstrætədʒi/` level.

OpenClaw 改变了这种状况。它作为你和所有代理之间的编排层——它将我的所有业务上下文（客户数据、会议记录、过去的决策、什么有效、什么失败）保存在我的 Obsidian 保险库中，并将历史上下文转换为每个编码代理的精确提示。代理专注于代码。编排器保持在高层战略层面。

Here's how the system works at a high level:

以下是系统的高级工作原理：

[![图像](https://pbs.twimg.com/media/HB0NSAEW0AAYPOF?format=jpg&name=medium)](/elvissun/article/2025920521871716562/media/2025790010293669888)

Last week Stripe wrote about their background agent system called "Minions" `/ˈmɪnjənz/` — parallel `/ˈpærəlel/` coding agents backed by a centralized orchestration layer. I accidentally built the same thing but it runs locally on my Mac mini.

上周 Stripe 写了关于他们的后台代理系统"Minions"的文章——由中央编排层支持的并行编码代理。我不小心构建了相同的东西，但它在我的 Mac mini 上本地运行。

Before I tell you how to set this up, you should know WHY you need an agent orchestrator.

在我告诉你如何设置这个系统之前，你应该知道为什么需要代理编排器。

## Why One AI Can't Do Both

## 为什么一个 AI 无法同时做两件事

Context windows are zero-sum. You have to choose what goes in.

上下文窗口是零和博弈。你必须选择放什么进去。

Fill it with code → no room for business context. Fill it with customer history → no room for the codebase. This is why the two-tier `/tɪr/` system works: each AI is loaded with exactly what it needs.

填满代码→没有业务上下文的空间。填满客户历史→没有代码库的空间。这就是为什么两层系统有效：每个 AI 都只加载它所需的内容。

OpenClaw and Codex have drastically `/ˈdræstɪkli/` different context:

OpenClaw 和 Codex 有截然不同的上下文：

[![图像](https://pbs.twimg.com/media/HB0EN2hXcAAbGi9?format=png&name=900x900)](/elvissun/article/2025920521871716562/media/2025780043406864384)

Specialization `/ˌspeʃələˈzeɪʃ(ə)n/` through context, not through different models.

通过上下文实现专业化，而不是通过不同的模型。

## The Full 8-step Workflow

## 完整的 8 步工作流程

Let me walk through a real example from last week.

让我带你回顾一个真实的上周案例。

### Step 1: Customer Request → Scoping with Zoe

### 第 1 步：客户请求 → 与 Zoe 进行范围界定

I had a call with an agency customer. They wanted to reuse configurations they've already set up across the team.

我与一家代理客户进行了通话。他们想在团队中重用已经配置好的配置。

After the call, I talked through the request with Zoe. Because all my meeting notes sync automatically to my obsidian `/əbˈsɪdiən/` vault `/vɔːlt/`, zero explanation `ˌekspləˈneɪʃ(ə)n/` was needed on my end. We scoped out the feature together — and landed `/ˈlændɪd/` on a template system that lets them save and edit their existing configurations.

通话后，我与 Zoe 讨论了请求。由于我所有的会议记录都自动同步到我的 Obsidian 保险库，我不需要做任何解释。我们一起界定了功能范围——决定采用一个模板系统，让他们可以保存和编辑现有的配置。

Then Zoe does three things:

然后 Zoe 会做三件事：

1.  Tops up credits to unblock customer immediately — she has admin API access

1.  为客户立即充值以解除阻塞——她有管理员 API 访问权限

2.  Pulls customer config from prod database — she has read-only prod DB access (my codex agents will never have this) to retrieve `/rɪˈtriːv/` their existing `/ɪɡˈzɪstɪŋ/` setup, which gets included in the prompt

2.  从生产数据库拉取客户配置——她有只读生产数据库访问权限（我的 codex 代理永远不会拥有这个）来检索他们现有的设置，该设置会包含在提示中

3.  Spawns a Codex agent — with a detailed prompt containing all the context

3.  生成一个 Codex 代理——包含所有上下文的详细提示


### Step 2: Spawn `/spɑːn/` the Agent

### 第 2 步：生成代理

Each agent gets its own worktree (isolated `/ˈaɪsəleɪtɪd/` branch) and tmux session:

每个代理都有自己的工作树（隔离的分支）和 tmux 会话：

```bash
# Create worktree + spawn agent
# 创建工作树 + 生成代理
git worktree add ../feat-custom-templates -b feat/custom-templates origin/main
cd ../feat-custom-templates && pnpm install

tmux new-session -d -s "codex-templates" \
  -c "/Users/elvis/Documents/GitHub/medialyst-worktrees/feat-custom-templates" \
  "$HOME/.codex-agent/run-agent.sh templates gpt-5.3-codex high"
```

The agent runs in a tmux session with full terminal logging via `/ˈvaɪə/` a script.

代理在 tmux 会话中运行，通过脚本进行完整的终端日志记录。

Here's how we launch `/lɔːntʃ/` agents:
这是我们启动代理的方式：

```bash
# Codex
codex --model gpt-5.3-codex \
  -c "model_reasoning_effort=high" \
  --dangerously-bypass-approvals-and-sandbox \
  "Your prompt here"

# Claude Code
claude --model claude-opus-4.5 \
  --dangerously-skip-permissions \
  -p "Your prompt here"
```

I used to use codex exec or claude -p, but switch to tmux recently:

我以前使用 codex exec 或 claude -p，但最近切换到了 tmux：

tmux is far better because mid-task redirection is powerful. Agent going the wrong direction? Don't kill it:

tmux 更好，因为任务中的重定向功能很强大。代理走错了方向？不要杀死它：

```bash
# Wrong approach:
# 错误的方法：
tmux send-keys -t codex-templates "Stop. Focus on the API layer first, not the UI." Enter

# Needs more context:
# 需要更多上下文：
tmux send-keys -t codex-templates "The schema is in src/types/template.ts. Use that." Enter
```

The task gets tracked in .clawdbot/active-tasks.json:

任务在 .clawdbot/active-tasks.json 中被跟踪：

```json
{
  "id": "feat-custom-templates",
  "tmuxSession": "codex-templates",
  "agent": "codex",
  "description": "Custom email templates for agency customer",
  "repo": "medialyst",
  "worktree": "feat-custom-templates",
  "branch": "feat/custom-templates",
  "startedAt": 1740268800000,
  "status": "running",
  "notifyOnComplete": true
}
```

When complete, it updates with PR number and checks. (More on this in step 5)

完成后，它会更新 PR 编号和检查状态。（更多内容在第 5 步）

```json
{
  "status": "done",
  "pr": 341,
  "completedAt": 1740275400000,
  "checks": {
    "prCreated": true,
    "ciPassed": true,
    "claudeReviewPassed": true,
    "geminiReviewPassed": true
  },
  "note": "All checks passed. Ready to merge."
}
```

### Step 3: Monitoring in a loop

### 第 3 步：循环监控

A cron `/ˈkrɑːn/` job runs every 10 minutes to babysit all agents. This pretty much functions as an improved Ralph `/rælf; reɪf/` Loop, more on it later.

一个 cron 作业每 10 分钟运行一次，来照看所有代理。这基本上像一个改进的 Ralph Loop，稍后会详细介绍。

But it doesn't poll the agents directly — that would be expensive. Instead, it runs a script that reads the JSON registry and checks:

但它不直接轮询代理——那会很昂贵。相反，它运行一个读取 JSON 注册表并检查的脚本：

```bash
.clawdbot/check-agents.sh
```

The script is 100% deterministic and extremely token-efficient:

该脚本是完全确定的并且非常节省 token：

- Checks if tmux sessions are alive
- 检查 tmux 会话是否存活
- Checks for open PRs on tracked branches
- 检查跟踪分支上的开放 PR
- Checks CI status via `gh cli`
- 通过 `gh cli` 检查 CI 状态
- Auto-respawns failed `/feɪld/` agents (max 3 attempts `/əˈtempts/`) if CI fails `/feɪlz/` or critical review feedback
- 如果 CI 失败或关键审查反馈，自动重新生成失败的代理（最多 3 次）
- Only alerts if something needs human attention
- 只有在需要人工关注时才发出警报

I'm not watching terminals. The system tells me when to look.

我不会盯着终端看。系统会告诉我什么时候需要查看。

### Step 4: Agent Creates PR

### 第 4 步：代理创建 PR

The agent commits, pushes, and opens a PR via `gh pr create --fill`. At this point I do NOT get notified — a PR alone isn't done.

代理提交、推送，并通过 `gh pr create --fill` 打开 PR。此时我不会收到通知——单个 PR 并不算完成。

Definition of done (very important your agent knows this):

完成定义（你的代理必须知道这一点，非常重要）：

- PR created - Branch synced to main (no merge conflicts)
- PR 已创建 - 分支已同步到主分支（没有合并冲突）
- CI passing (lint, types, unit tests, E2E)
- CI 通过（lint、类型检查、单元测试、端到端测试）
- Codex review passed
- Codex 审查通过
- Claude Code review passed
- Claude Code 审查通过
- Gemini review passed
- Gemini 审查通过
- Screenshots included (if UI changes)
- 包含截图（如果 UI 有变化）

### Step 5: Automated Code Review

### 第 5 步：自动化代码审查

Every PR gets reviewed by three AI models. They catch different things:

每个 PR 都由三个 AI 模型审查。它们发现不同的问题：

*   Codex Reviewer — Exceptional `/ɪkˈsepʃən(ə)l/` at edge cases. Does the most thorough `/ˈθɜːroʊ/` review. Catches logic errors, missing error handling, race conditions. False positive rate is very low.
*   Codex 代码审查工具 — 擅长处理边界情况。提供最全面的代码审查，能发现逻辑错误、缺失的错误处理和竞争条件。误报率极低。

*   Gemini Code Assist Reviewer — Free and incredibly `/ɪnˈkredəbli/` useful. Catches security issues, scalability problems other agents miss. And suggests specific fixes. No brainer to install.

*   Gemini 代码辅助审查工具 — 免费且极其实用。能发现其他工具遗漏的安全问题和可扩展性问题，并给出具体修复建议。安装毫无疑问。

*   Claude Code Reviewer — Mostly useless - tends to be overly cautious `ˈkɔːʃəs/`. Lots of "consider adding..." suggestions that are usually overengineering. I skip everything unless it's marked critical. It rarely `/ˈrerli/` finds critical issues on its own but validates what the other reviewers flag.

*   Claude代码审查工具 — 基本无用——过度谨慎倾向明显。大量“建议添加...”提示往往属于过度设计。除非标记为关键问题，否则我直接忽略所有建议。该工具极少独立发现关键问题，但能验证其他审查工具标记的缺陷。



All three post comments directly on the PR.

所有三个都会在 PR 上直接发表评论。

### Step 6: Automated Testing

### 第 6 步：自动化测试

Our CI pipeline runs a heavy amount of automated tests:

我们的 CI 管道运行大量自动化测试：

- Lint and TypeScript checks 
- Lint 和 TypeScript 检查
- Unit tests
- 单元测试
- E2E tests
- 端到端测试
- Playwright tests against a preview environment (identical to prod)
- 针对预发布环境的 Playwright 测试（与生产环境相同）

I added a new rule last week: if the PR changes any UI, it must include a screenshot `/ˈskriːnʃɑːt/` in the PR description. Otherwise CI fails. This dramatically `/drəˈmætɪkli/` shortens `/ˈʃɔːrt(ə)n/` review time — I can see exactly what changed without clicking through the preview.

我上周添加了一条新规则：如果 PR 更改了任何 UI，必须在 PR 描述中包含截图。否则 CI 失败。这大大缩短了审查时间——我无需点击预览就能准确看到变化。

### Step 7: Human Review

### 第 7 步：人工审查

Now I get the Telegram notification: "PR #341 ready for review."

现在我收到 Telegram 通知："PR #341 准备审查。"

By this point:

到此时：

- CI passed
- CI 通过
- Three AI reviewers approved the code
- 三个 AI 审查者批准了代码
- Screenshots show the UI changes
- 截图显示了 UI 变化
- All edge cases are documented in review comments
- 所有边缘情况都在审查评论中记录

My review takes 5-10 minutes. Many PRs I merge without reading the code — the screenshot shows me everything I need.

我的审查需要 5-10 分钟。许多 PR 我不阅读代码就合并——截图向我展示了所需的一切。

### Step 8: Merge

### 第 8 步：合并

PR merges. A daily cron job cleans up orphaned `/ˈɔːf(ə)nd/` worktrees and task registry json.

PR 合并。每日 cron 作业清理孤立的工作树和任务注册表 json。

## The Ralph Loop V2

## Ralph Loop V2

This is essentially `/ɪˈsenʃəli/` the Ralph Loop, but better.

这本质上就是 Ralph Loop，但更好。

The Ralph Loop pulls context from memory, generate output, evaluate `/ɪˈvæljueɪt/` results, save learnings. But most implementations `/ˌɪmplɪmenˈteɪʃ(ə)n/` run the same prompt each cycle `/ˈsaɪk(ə)l/`. The distilled `/dɪˈstɪld/` learnings improve future retrievals `/rɪˈtriːv(ə)l/`, but the prompt itself stays static.

Ralph Loop 从内存中提取上下文，生成输出，评估结果，保存学习成果。但大多数实现每轮运行相同的提示。提炼的学习成果改善了未来的检索，但提示本身保持静态。

Our system is different. When an agent fails `/feɪlz/`, Zoe doesn't just respawn it with the same prompt. She looks at the failure `/ˈfeɪljər/` with full `/fʊl/` business context and figures `/ˈfɪɡjərz/` out how to unblock it:

我们的系统不同。当代理失败时，Zoe 不会用相同的提示重新生成它。她会查看带有完整业务上下文的失败情况，并找出如何解除阻塞：

*   Agent ran out of context? "Focus only on these three files."
*   Agent 偏离了上下文？“只关注这三个文件。”

*   Agent went the wrong direction? "Stop. The customer wanted X, not Y. Here's what they said in the meeting."
*   Agent方向错误？“停下。客户想要的是X，不是Y。这是他们在会议中的原话。”

*   Agent need clarification? "Here's customer's email and what their company does."
*   Agent需要澄清？“这是客户的邮件及其公司业务简介。”

Zoe babysits agents through to completion `/kəmˈpliːʃ(ə)n/`. She has context the agents don't — customer history, meeting notes, what we tried before, why it failed. She uses that context to write better prompts on each retry.

Zoe 照看代理直到完成。她拥有代理没有的上下文——客户历史、会议记录、我们之前尝试过的内容、失败的原因。她使用这些上下文在每次重试时编写更好的提示。

But she also doesn't wait for me to assign tasks. She finds work proactively:

但她也不会等待我分配任务。她会主动寻找工作：

*   Morning: Scans Sentry `/ˈsentri/` → finds 4 new errors → spawns 4 agents to investigate and fix
*   早晨：扫描Sentry → 发现4个新错误 → 生成4个代理进行调查修复

*   After meetings: Scans meeting notes → flags 3 feature requests customers mentioned → spawns 3 Codex agents
*   会议后：扫描会议记录 → 标记客户提及的3个功能需求 → 生成3个Codex代理

*   Evening: Scans git log → spawns Claude Code to update changelog and customer docs
*   傍晚：扫描Git日志 → 生成Claude代码更新变更日志和客户文档

I take a walk after a customer call. Come back to Telegram: "7 PRs ready for review. 3 features, 4 bug fixes."

我在客户通话后散步。回到 Telegram 时："7 个 PR 准备审查。3 个功能，4 个错误修复。"

When agents succeed `/səkˈsiːd/`, the pattern gets logged. "This prompt structure works for billing features." "Codex needs the type definitions `/ˌdefɪˈnɪʃnz; ˌdefəˈnɪʃənz/` upfront `/ˌʌpˈfrʌnt/`." "Always include the test file paths."

当代理成功时，模式会被记录下来。"这个提示结构适用于计费功能。" "Codex 需要前置的类型定义。" "总是包含测试文件路径。"

The reward signals are: CI passing, all three code reviews passing, human merge. Any failure triggers the loop. Over time, Zoe writes better prompts because she remembers what shipped.

奖励信号是：CI 通过、所有三个代码审查通过、人工合并。任何失败都会触发循环。随着时间的推移，Zoe 会编写更好的提示，因为她记得哪些功能已发布。

## Choosing the Right Agent

## 选择合适的代理

Not all coding agents are equal. Quick reference:

并非所有编码代理都相等。快速参考：

Codex is my workhorse `/ˈwɜːrkhɔːrs/`. Backend logic, complex bugs, multi-file refactors, anything that requires reasoning across the codebase. It's slower but thorough `/ˈθɜːroʊ/`. I use it for 90% of tasks.

Codex 是我的主力。后端逻辑、复杂错误、多文件重构、任何需要跨代码库推理的任务。它比较慢但很彻底。我 90% 的任务都使用它。

Claude Code is faster and better at frontend work. It also has fewer permission issues, so it's great for git operations. (I used to use this more to drive day to day, but Codex 5.3 is simply better and faster now)

Claude Code 在前端工作中更快、更好。它权限问题较少，因此非常适合 git 操作。（我以前更多地使用它来驱动日常工作，但现在 Codex 5.3 明显更好更快）

Gemini has a different superpower — design sensibility `/ˌsensəˈbɪləti/`. For beautiful UIs, I'll have Gemini generate an HTML/CSS spec first, then hand that to Claude Code to implement in our component `/kəmˈpoʊnənt/` system. Gemini designs, Claude builds.

Gemini 有不同的超能力——设计感。对于美观的 UI，我会让 Gemini 先生成 HTML/CSS 规范，然后将其交给 Claude Code 在我们的组件系统中实现。Gemini 设计，Claude 构建。

Zoe picks the right agent for each task and routes `/ruːts/` outputs between them. A billing system bug goes to Codex. A button style fix goes to Claude Code. A new dashboard design starts with Gemini.

Zoe 为每个任务选择合适的代理，并在它们之间路由输出。计费系统错误交给 Codex。按钮样式修复交给 Claude Code。新仪表板设计从 Gemini 开始。

## How to Set This Up

## 如何设置

Copy this entire article into OpenClaw and tell it: "Implement this agent swarm setup for my codebase."

将整篇文章复制到 OpenClaw 并告诉它："为我的代码库实现这个代理群设置。"

It'll read the architecture, create the scripts, set up the directory `/dəˈrektəri/` structure, and configure cron monitoring. Done in 10 minutes.

它会阅读架构、创建脚本、设置目录结构、配置 cron 监控。10 分钟内完成。

No course to sell you.

没有课程要卖给你。

## The Bottleneck Nobody Expects

## 无人预料到的瓶颈

Here's the ceiling I'm hitting right now: RAM.

这是我目前遇到的上限：RAM。

Each agent needs its own worktree. Each worktree needs its own `node_modules`. Each agent runs builds, type checks, tests. Five agents running simultaneously `/ˌsaɪm(ə)lˈteɪniəsli/` means five parallel `/ˈpærəlel/` TypeScript compilers, five test runners, five sets of dependencies `/dɪˈpendənsiz/` loaded into memory.

每个代理都需要自己的工作树。每个工作树都需要自己的 `node_modules`。每个代理都运行构建、类型检查、测试。五个代理同时运行意味着五个并行的 TypeScript 编译器、五个测试运行器、五组依赖项加载到内存中。

My Mac Mini with 16GB tops out at 4-5 agents before it starts swapping `/ˈswɒpɪŋ/` — and I need to be lucky they don't try to build at the same time.

我的 16GB Mac Mini 最多支持 4-5 个代理就会开始交换——我需要幸运地让它们不会同时尝试构建。

So I bought `/bɔːt/` a Mac Studio M4 max with 128GB RAM ($3,500) to power this system. It arrives end of March and I'll share if it's worth it.

所以我买了一台配备 128GB RAM 的 Mac Studio M4 max（$3,500）来驱动这个系统。它将于 3 月底到达，我会分享它是否值得。

## Up Next: The One-Person Million-Dollar Company

## 下一步：一个人的百万美元公司

We're going to see a ton `/tʌn/` of one-person million-dollar companies starting in 2026. The leverage `/ˈlevərɪdʒ/` is massive `/ˈmæsɪv/` for those who understand how to build recursively `/ri'kəsivli/` self-improving agents.

我们将看到 2026 年开始大量出现的一人百万美元公司。对于那些懂得如何构建递归自我改进代理的人来说，杠杆效应是巨大的。

This is what it looks like: an AI orchestrator `/ˈɔːrkɪstreɪt/` as an extension `/ɪkˈstenʃ(ə)n/` of yourself (like what Zoe is to me), delegating work to specialized `/ˈspeʃəlaɪzd/` agents that handle different business functions. Engineering. Customer support. Ops `/ɑːps/`. Marketing. Each agent focused on what it's good at. You maintain `/meɪnˈteɪn/` laser `/ˈleɪzər/` focus and full control.

这就是它的样子：AI 编排器作为你自己的延伸（就像 Zoe 对我一样），将工作委托给处理不同业务功能的专门代理。工程、客户支持、运营、营销。每个代理都专注于其擅长的领域。你保持激光般的专注和完全控制。

The next generation `/ˌdʒenəˈreɪʃ(ə)n/` of entrepreneurs `/ˌɑːntrəprəˈnɜːr/` won't hire `/ˈhaɪər/` a team of 10 to do what one person with the right system can do. They'll build like this — staying small, moving fast, shipping daily.

下一代企业家不会雇佣一个 10 人的团队来做一个拥有合适系统的一个人就能完成的工作。他们会这样构建——保持小规模、快速移动、每日发布。

There's so much AI-generated slop `/slɑːp/` right now. So much hype `/haɪp/` around agents and "mission controls" without building anything actually useful. Fancy `/ˈfænsi/` demos with no real-world benefits `/ˈbenɪfɪts/`.

现在有很多 AI 生成的垃圾。关于代理和"任务控制"有很多炒作，但没有构建任何真正有用的东西。花哨的演示，没有实际的好处。

I'm trying to do the opposite `/ˈɑːpəzɪt/`: less hype, more documentation `/ˌdɑːkjumenˈteɪʃn/` of building an actual business. Real customers, real revenue `/ˈrevənuː/`, real commits that ship to production, and real loss too.

我试图做相反的事情：减少炒作，更多地记录构建真实业务的过程。真实的客户、真实的收入、真实发布到生产的提交，以及真实的损失。

What am I building? Agentic PR — a one-person company taking on the enterprise `/ˈentərpraɪz/` PR incumbents `/ɪnˈkʌmbənts/`. Agents that help startups get press coverage without a $10k/month retainer `rɪˈteɪnər/`.

我在构建什么？代理式 PR——一个挑战企业 PR 老牌选手的一人公司。帮助初创公司获得媒体报道的代理，无需支付每月 10,000 美元的费用。

If you want to see how far I take this, follow along.

如果你想看我走到多远，请关注我。