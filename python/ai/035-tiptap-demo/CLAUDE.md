# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目约定（先读这两份）

- 启动时必读 `README.md`（前端）和 `server/README.md`（后端）—— 细节都在那里面。
- 做完重要改动后更新这两个文件。
- 本项目不需要你打开浏览器测试，用户自己测。
- **不写新的自动化测试** —— 现有测试保持绿就行；新功能靠手测 + 调试日志验证。
- **加调试日志**（`server/.env` 的 `DEBUG=1` 开关，server 端 + 浏览器 console 共用），
  写完代码告诉用户怎么测，用户会把日志贴回来。日志前缀统一 `[debug]` / `[auth]` / `[usage]` / `[llm]`。
- 用 pnpm，不用 npm。

## 仓库结构

这是**两个独立项目**，各有自己的 `package.json` / `pnpm-lock.yaml`：

- 根目录 = 前端：Vue 3 + Vite + vue-router + Tiptap 3，端口 5173。
- `server/` = 后端：Express + TypeScript（tsx 跑）+ vitest，端口 8787。

Vite 把 `/api` 代理到 `http://127.0.0.1:8787`，前端一律 `fetch('/api/...')`。前后端必须同时起。

## 常用命令

```bash
# 前端（根目录）
pnpm install
pnpm dev            # vite, http://127.0.0.1:5173
pnpm build          # vue-tsc --noEmit && vite build
pnpm type-check     # vue-tsc --noEmit
```

前端**没有** test / lint 脚本。

```bash
# 后端（server/）
cd server
pnpm install
pnpm dev            # tsx watch src/index.ts, http://127.0.0.1:8787
pnpm test           # vitest run（全部）
pnpm test:watch
pnpm type-check     # tsc --noEmit
```

跑单个测试文件 / 单个用例（vitest 参数直接跟在脚本名后面）：

```bash
cd server
pnpm test tests/translate.test.ts
pnpm test tests/translate.test.ts -t "英译中"
pnpm exec vitest run tests/dictionary.test.ts   # 同样可用
```

后端手工冒烟（需要 Ollama 或已配好的 provider 在跑）：

```bash
cd server
bash curl-tests.sh              # 全套
bash curl-tests.sh translate    # 单个分组
BASE=http://host:port bash curl-tests.sh
```

> curl 走「本地工具调用」路径（`AUTH_LOCAL_BYPASS=1` 时免登录），所以脚本不用改。

后端 CLI —— 不起 server 直接调 service 层，排错首选：

```bash
cd server
pnpm cli --help
pnpm cli sentence --text "..." --operation rewrite --tone formal
pnpm cli selection --text "very important" --kind phrase --context "..."
pnpm cli grammar --text "He go to school last yesterday."
pnpm cli translate --to zh --text "Education is very important for children."
pnpm cli models
```

调试开关：`--prompt` 打印**真实发出**的 system/user，`--raw` 打印模型原始输出绕过解析，`--json` 一次性 dump。

## 前端架构要点

- **外壳 + 嵌套路由**：`App.vue` = `AppSidebar` + `<RouterView>`；子路由 `/`（编辑器）、`/settings`（设置页）。侧栏在两个页面常驻。**`/login` 是顶层路由**（不套外壳）—— 必须排在 `'/'` 之前，否则会被父路由吃掉。
- **登录守卫**：`router.beforeEach` 里 `await loadAuth()`，没登录就跳 `/login`。`useAuth` 是模块级单例，**不 import router**（避免 `router → pages → services → useAuth → router` 的循环）—— 跳转逻辑放在 `router.ts` 里。
- **跨组件状态用模块级单例 composable**（`useMode` / `useTones` / `useAuth`），不是 Pinia。原因是侧栏在 `App.vue`、编辑器在 `EditorPage.vue`，不是父子关系；状态同时写 localStorage（刷新保留）。
- **双栏方向**：两个编辑器的 DOM 位置固定，靠 CSS `order` 排源/译 —— 切方向不重建编辑器，内容不丢。`mode` 只有 `zh2en` / `en2zh` 两个值。
- **句子对齐是双栏联动的唯一依据**：`useWritingContext.ts` 按标点切句（中文 `。！？.!?`、英文 `.!?`，换行也算边界），`sentenceIndex` 是**文档级全局索引**（跨段落累计）。源侧和译侧的句子必须**数量相同、顺序一致** —— 任何一边多句/少句，高亮和「对应中文原文」就全错位。
- **选区分类**：`useSelectionKind.getSelectionInfo()` 返回 `word` / `phrase` / `sentence`，跨句跨段或只有光标返回 `null`（不弹菜单）。工具栏那排常驻按钮用 `getSentenceAtCursor()`，作用于光标所在句，不需要选中字符。
- **浮动菜单定位**（`AiContextMenu.vue`）：`editor.view.coordsAtPos()` 取选区上下两条 client rect → `Teleport` 到 body → `position: fixed` 定位。**定位必须按菜单实测高度算**（`ResizeObserver` 盯高度），不能用估计值，否则查词词条 / 语法结果卡片会被裁到视口外。菜单整体在 `mousedown` 上 `preventDefault`，否则 ProseMirror 先吃掉选区，点候选时 range 已经没了。
- **API 调用**统一走 `src/services/http.ts`（`http.get/post/put/del`），它会把后端 `{error}` 字段捞成 `ApiError`。`services/translate.ts` 是例外（自带 `pLimit` 并发 3）。
- 英译中模式下英文侧 `aiEnabled=false` —— 改写 / 润色 / 语法 / 指令全部关掉，只留「查词」。

## 后端架构要点

分层：`config/`（配置 + 模板 + 渲染）→ `llm/`（provider 客户端）→ `services/`（能力编排）→ `routes/`（校验 + 状态码分类）。

**所有持久化都在一个 SQLite 库**（`server/data/app.db`，用 Node 内置 `node:sqlite`，零第三方依赖）：
配置（providers / app_state / prompt_templates / prompt_tones / prompt_versions）+ 用户 / 会话 / token 用量。
`server/data/config.json` 现在只是**首次导入源** —— 库里没 provider 时自动导入，之后不再回写。

**提示词全部可配**（存在 SQLite 里），每个 provider 一套完整独立的提示词集。执行链路：

```
调用参数 → capabilities.ts buildContext() → template.ts render()（mustache-lite）
        → llm client chat() → capabilities.ts extractOutput() → 响应
```

- **模板渲染器**在 `config/template.ts`（自研 mustache-lite，约 220 行，无第三方依赖）。支持 `{{var}}` / `{{#sec}}` / `{{^sec}}` / `{{! 注释}}` / `{{a.b}}`，**section 标签独占一行时要裁掉整行**（standalone 行处理）—— 渲染结果必须和迁移前的 builder 逐字节相同。
- **提示词文本在配置里，解析逻辑仍在代码里**（`src/prompts/`）：`cleanText`（剥围栏/前缀/引号）、`parseCandidates`（整段 JSON → 围栏 JSON → 正文里的 `[...]`）。脏活是配置表达不了的。
- **`grammarCheck` 不走 `cleanText`** —— 它的输出是给用户看的原文，模型写什么就透传什么。
- **`tone` 模板变量装的是整行指令**（`Write in a formal tone: …`）；不传 tone 时 `{{#tone}}` 整块跳过，输出必须与加语气之前逐字节一致。
- **语气 id 的唯一来源**是 `server/src/config/defaults.ts` 的 `TONE_IDS`，前端 `src/constants/tones.ts` 要与之保持一致。每个 provider 的 label / guidance 各自独立。
- **LLM 抽象**：`llm/types.ts` 的 `LlmClient` 接口，三家客户端（`ollamaClient` / `openaiClient` / `anthropicClient`）实现同一形状。结构化输出的降级链**只在 HTTP 400 时触发**，其它错误直接抛。
- **三个客户端返回前都调 `stripReasoning()`** —— 模型（尤其 MiniMax-M3 走 OpenAI 兼容端点）会把 `<think>…</think>` 混进 `content`，不剥的话语法检查会把整段思考显示给用户、翻译/改写解析直接失败。
- **DI**：`createApp({ aiService, translateService, configStore, registry, dictionaryService, userStore?, recordUsage? })`，测试注入 mock 打真实路由层，不需要起真模型。**`userStore` 不传 = 不启用鉴权**（单元测试靠这个绕开登录）。
- **状态码分类**统一在 `routes/httpError.ts` 的 `classifyError()`：400 入参 / 404 不存在 / 500 解析失败 / 502 上游失败 / 503 配置不全。
- **查词纯本地**（`services/dictionary.ts`）：读 JSONL 词库，用**字节偏移索引**（建索引时整文件读成 Buffer 按 `\n` 扫，只留 `Map<wordHead, offset>`，查询时 seek + readline 解析那一行）。词库只收词头，查变形会退一步还原。

## 鉴权与用量记账（新增）

- **会话**：httpOnly cookie（`sid`，30 天）+ `sessions` 表。密码用 `node:crypto` 的 `scryptSync`（零依赖），库里存 `saltHex:hashHex`。
- **消耗 token 的接口必须登录**（`routes/auth.ts` 的 `requireUser`）：`/api/translate`、`/api/rewrite-*`、`/api/providers/:id/test`、`.../dry-run`。
- **本地工具免登录**：浏览器经 Vite 代理过来时后端看到的 IP 也是 127.0.0.1，所以**不能只看 IP** —— 靠浏览器特有的 `Origin` / `Sec-Fetch-Site` 头区分。回环 + 无这两个头 = 本地工具（curl / CLI）→ 放行但不记账。`.env` 的 `AUTH_LOCAL_BYPASS=0` 可关。
- **用量记账**：`ChatResult.usage` 由三个客户端填充（Ollama `prompt_eval_count`/`eval_count`；OpenAI `prompt_tokens`/`completion_tokens`；Anthropic `input_tokens`/`output_tokens` + cache 两段）→ service 层在 `chat()` 成功后、`extractOutput()` **之前**记 → 落到 `token_usage` 表（每人每天一行，UPSERT 累加）。
- **「记到谁头上」用 AsyncLocalStorage**（`services/requestContext.ts`）：登录守卫 `runAsUser(user, next)` 把身份挂进请求上下文 —— 并发安全，且**不用改任何 service 的方法签名**。
- **`day` 列是 INTEGER（YYYYMMDD）**，不是 TEXT —— SQLite 没有真正的 DATE 类型，INTEGER 比较没有 affinity 歧义。日/月边界按**服务端本地时区**算。
- CLI 不经过 HTTP → 不计入任何用户。

## 调试日志

`server/.env` 的 `DEBUG=1` 一个开关同时控制 server 端（`utils/debug.ts` 的 `debugLog`）和浏览器端（前端通过 `GET /api/debug` 拿到开关后给 `window.fetch` 打补丁）。**前端不单独配 .env。**

## 配置

- `server/.env`（**不进 git**）：`PORT` / `OLLAMA_HOST` / `OLLAMA_MODEL` / `MAX_CANDIDATES` / `<PROVIDER_ID>_KEY` / `VOCAB_PATH` / `DB_PATH` / `ADMIN_USERNAME` / `ADMIN_PASSWORD` / `AUTH_LOCAL_BYPASS` / `DEBUG`。启动时自动加载（`config/env.ts`），`SERVER_ENV_FILE` 可换文件。
- `server/data/app.db`（**不进 git**）：配置 + 用户 + 会话 + token 用量。库里**永远没有明文 key** —— provider 的 `apiKeyEnv` 只记变量名。
- 切换生效 provider **即时生效**，不用重启后端（服务层每次请求都重新问 `activeProvider.ts` 的 getter）。
- 改 `VOCAB_PATH` 后可在设置页「词库」tab 点「重新加载」，不用重启。

## 测试约定

后端用 `vitest` + `supertest`，mock 从 `tests/helpers.ts` 注入。

**`tests/promptParity.test.ts` 是迁移安全网**：默认提示词集的渲染结果必须和迁移前的 builder 逐字节相同。它的期望值**不能手抄** —— 期望值取自 `tests/legacy/prompts/` 里冻结的原始实现（**不要改那个目录**）。手抄的期望值会和模板一起错，测试反而是绿的。

其余测试文件覆盖见 `server/README.md` 的测试表格。

## 容易踩的坑

- 后端库里 provider 的 `apiKeyEnv` 存的是**变量名**；`PUT /api/providers` 收 `apiKey` 会直接 400。
- `structuredOutput` 的 `temperature` 对 Anthropic 无效（API 已移除采样参数）；Anthropic 的 `max_tokens` 是必填，且是「思考 + 回复」的共同上限（默认 16000）。
- OpenAI 的 `strict: true` 只支持 JSON Schema 子集 → 代码里固定 `strict: false`。
- 前端浮动菜单的二级面板在**菜单内部展开**，不是飞出式子菜单（菜单 `overflow: hidden`，飞出式会被裁掉）。
- 后端 `src/` 下所有 import 都带 `.js` 后缀（ESM + tsx）。
- **`node:sqlite` 必须用 `createRequire` 加载**（`config/db.ts`）：Vite 5 的内置模块列表里还没有 `sqlite`，直接 `import from 'node:sqlite'` 会被打包器当成 npm 包去解析（`Failed to load url sqlite`）—— vitest 和前端构建都会炸。
- 库开了 WAL，会生成 `app.db-wal` / `app.db-shm` 两个附属文件 —— `server/.gitignore` 里是 `data/*.db*`。
- 后端启动日志**只打印账号名，不回显密码**（密码在 `server/.env` 的 `ADMIN_PASSWORD` 里）。
