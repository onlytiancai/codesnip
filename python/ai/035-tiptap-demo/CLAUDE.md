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
- **登录守卫**：`router.beforeEach` 里 `await loadAuth()`，没登录就跳 `/login`。`useAuth` 是模块级单例，**不 import router**（避免 `router → pages → services → useAuth → router` 的循环）—— 跳转逻辑放在 `router.ts` 里。标签页标题在 `router.afterEach` 里按路由名设（`TITLES` 表）。
- **侧栏底部**：**用量在上、用户块在下**。用户块 = 用户名 + 套餐徽标，**点它弹菜单**（设置 / 退出登录），菜单用 `position: absolute` 相对 `.sidebar__user-wrap` **向上弹**（用户块贴着底边）。点外面 / Esc / 换路由都收起。
- **跨组件状态用模块级单例 composable**（`useMode` / `useTones` / `useAuth`），不是 Pinia。原因是侧栏在 `App.vue`、编辑器在 `EditorPage.vue`，不是父子关系；状态同时写 localStorage（刷新保留）。
- **双栏方向**：两个编辑器的 DOM 位置固定，靠 CSS `order` 排源/译 —— 切方向不重建编辑器。**两个方向各存一份两栏内容**（`EditorPage.vue` 的 `modeHtml`）：切方向时把旧方向的存下、装上新方向的 —— 中译英里输入的中文不会带进英译中，各自独立。装内容期间用 `restoring` 标志屏蔽「源侧清空 → 译侧跟着清」和实时翻译。`mode` 只有 `zh2en` / `en2zh` 两个值。
- **句子对齐是双栏联动的唯一依据**：`useWritingContext.ts` 按标点切句（中文 `。！？.!?`、英文 `.!?`，换行也算边界），`sentenceIndex` 是**文档级全局索引**（跨段落累计）。源侧和译侧的句子必须**数量相同、顺序一致** —— 任何一边多句/少句，高亮和「对应中文原文」就全错位。
  **点不是见到就断句**：`PHP8.4` / `3.14` / `www.a.com` / `Mr. Smith` / `e.g.` / `U.S.` 里的点都不是句号 —— 规则：点后面（跳过右括号引号）得是空白、中日韩字符或结尾，且点前面那个词不能是缩写。
- **选区分类**：`useSelectionKind.getSelectionInfo()` 返回 `word` / `phrase` / `sentence`，跨句跨段或只有光标返回 `null`（不弹菜单）。工具栏「✨ AI」下拉里的那几项用 `getSentenceAtCursor()`，作用于光标所在句，不需要选中字符。
- **点词即候选**（只中译英的英文框，`aiEnabled` 时才开）：光标停在某个词的**最左侧**、或鼠标点在这个词上 → 直接弹这个词的**候选词**面板（不用先拖选），点候选词就替换掉原词。找词用 `useSelectionKind.getWordInfoAt(editor, pos, mode)`，`mode` = `'leftEdge'`（只认词首，词在光标右边）/ `'any'`（点击取整个词，词内撇号算一个词）。
  ⚠️ ProseMirror 的 `handleClick` 跑在**选区更新之前**（mouseup 里先调它再更新选区）—— 点击那条路必须用它给的 `pos`；选区更新那条路要用 `lastClickAt`（500ms 内）让开，否则「词首」规则会把点击的结果顶掉。另外打字时 transaction 改了文档，要靠 `transaction.docChanged` 挡掉，不然每敲一个字都会弹一次。
  ⚠️ **菜单不许「闪」**：点词时菜单要**原地换内容**，不能先关掉再弹出来。所以 `onShellMousedownCapture` 遇到「点在词上」时**不关菜单**，`detectSelection` 遇到「只有光标 + 刚点过鼠标」时也**不关**。
- **工具栏的 AI 指令收在下拉里**（`TiptapEditor.vue`）：工具栏只留一个「✨ AI」按钮，点开才是 6 项清单。下拉**必须 Teleport 到 body** —— `.editor-shell` 是 `overflow: hidden`，放在工具栏里会被裁掉；用 `position: fixed` + 按钮的 `getBoundingClientRect()` 定位（右对齐、下方放不下就往上翻）。点外面 / Esc / 选一项都收起。
  ⚠️ 工具栏按钮的样式要用 `.toolbar .xxx` **复合选择器**：`.toolbar button` 的特异性比单个类高，只写 `.toolbar__btn-copy` 会被盖掉。
- **浮动菜单定位**（`AiContextMenu.vue`）：`editor.view.coordsAtPos()` 取选区上下两条 client rect → `Teleport` 到 body → `position: fixed` 定位。**定位必须按菜单实测高度算**（`ResizeObserver` 盯高度），不能用估计值，否则查词词条 / 语法结果卡片会被裁到视口外。菜单整体在 `mousedown` 上 `preventDefault`，否则 ProseMirror 先吃掉选区，点候选时 range 已经没了。
- **悬浮框的结构**（`AiContextMenu.vue`）：选中**词 / 短语**时顶栏是**两个 tab** ——「其它候选单词」（默认落在这个 tab，直接给候选词）/「查词」；选中**整句**时是菜单项清单（改写语气 / 润色 / 精简 / 扩写）。**语法检查、自定义指令已从悬浮框移除**（工具栏「✨ AI」下拉里还有）。英译中（`allowAi=false`）不出现 tab，只剩「📖 查词」。
  ⚠️ 两个 `immediate: true` 的 watcher（选区的、autoAction 的）**都放 setup 末尾**，且**选区的那个在前** —— 它会 `resetState()`（顺手 `abort()` 在飞的请求），反过来会把 autoAction 刚发出去的请求掐掉。选区那个还会摸 `candidatesTimer`，写在它声明前面就是 TDZ。
  ⚠️ 候选词请求**防抖 250ms**：拖选时选区每挪一下就变一次，不防抖的话每变一次发一次请求。
- **自定义指令（用户提示词）**：`composables/useCustomPrompts.ts` 是模块级单例（当前用户的指令 + 跟着**当前生效的 provider** 走）。
  英文编辑器的工具栏（`toolbar__btn-prompt`，最多 5 个）和「✨ AI ▾ → 我的指令」都从它取。
  ⚠️ **作用范围**决定改哪一段（`resolveCustomTarget()`）：当前句 → text=光标所在句、context=所在段落；当前段 → text=所在段、context=全文摘要；所有正文 → text=整篇正文。
  跑指令走 `POST /api/custom-prompts/:id/run`（**指令文本从库里取**，客户端只送 text/context/sourceSentence）→ 结果卡片复用 `AiContextMenu.vue` 的 `customResult` prop（`✓接受/✕拒绝/继续调整`）。
  ⚠️ 接受时按范围替换：句子/段落用 `insertContentAt` 局部替换，**「所有正文」要整篇 `setContent`**（按换行拆回段落，别把结构拍平）。
- **API 调用**统一走 `src/services/http.ts`（`http.get/post/put/del`），它会把后端 `{error}` 字段捞成 `ApiError`。`services/translate.ts` 是例外（自带 `pLimit` 并发 3）。
- **实时翻译**（`EditorPage.vue`，默认关，只中译英方向）：中文改了停手 800ms 自动翻，**以句子为单位** —— `liveCache`（源句→译文，语气算进 key）复用没改过的句子，只翻新增/改动的；写回时 `TiptapEditor.applyParagraphs()` **只替换变了的那几段**（从后往前改，避免前面段落的 doc 位置被带偏），整篇 `setContent` 会让译文框闪一下、滚动跳回顶部。段落数变了才整体重建。
- **手动翻译是流式的**（`EditorPage.vue` 的 `doTranslate`）：走 `services/translate.ts` 的 `translateSentencesStream`，**每句一回调**。译侧先铺一层占位（每句一个 `…`），翻完一句就把那一段重写一次（`applyParagraphs`，只改一段）；源侧正在翻的句子用 Decoration 画 `.sentence-loading`（琥珀色脉冲 + 句尾转圈，伪元素不进编辑器内容），失败的画 `.sentence-failed`。
- **loading 索引 = 源侧的全局句子索引**：跟 `collectSentences`（`useWritingContext.ts`）同一套编号（跨段落累计），所以装饰能直接对上。索引来源是 `extractParagraphs` + 切句后的**摊平下标** —— 中文框只有段落（heading/list 都禁了），两者对得上；英文框有列表时列表里的句子不参与编号（老问题）。
- **翻译期间锁编辑器**：两个编辑器组件都有 `editable` prop → `editor.setEditable(false)`；成功/失败都在 `finally` 里解锁。整体失败（401/403）时译侧还原回原来的内容。`loadingOwner` 标记谁在占用 loading 状态 —— 手动翻译会掐掉实时翻译，没有这个标记会互相清空。
- **中文框粘贴去格式**：`ChineseEditor.vue` 的 `editorProps.transformPastedHTML` → `utils/html.ts` 的 `htmlToPlainParagraphs`（DOMParser 取纯文本 + 段落划分，加粗/代码块/引用/列表全丢）。只对中文框生效。
- **英文框悬停气泡**：`TiptapEditor.vue` 的 mousemove → `posAtCoords` → `getWordInfoAt` → 用 `aligned` 对齐表查这句的中文原文；rAF 节流，AI 菜单开着时不弹（免得两个浮层叠一起）。装饰 `.hover-word`。
- **工具栏的清空/复制跟着源译侧走**：两个编辑器组件都有 `isSource` prop —— **源侧渲染「清空」，译侧渲染「复制」**，切方向时自动互换（中译英：中文=清空 / 英文=复制；英译中反过来）。复制逻辑抽在 `composables/useCopyContent.ts`（纯文本提取 + 剪贴板 + 「✓ 已复制」提示），两个框共用。
- **源侧清空 → 译侧跟着清**：译文没意义了，`EditorPage.vue` 里清掉译侧和对齐表。
  ⚠️ **不能 `watch(sourceHtml)`**（它是随 mode 变的 computed）—— 切方向的瞬间新方向的源侧通常是空的，会把另一栏的用户内容误清掉。要**分别 watch `zhHtml` / `enHtml`**，再按当前 `sourceLang` 判断该不该联动。
- 英译中模式下英文侧 `aiEnabled=false` —— 改写 / 润色 / 语法 / 指令全部关掉，只留「查词」。
- **设置页的 tab**（`SettingsPage.vue`）：LLM 供应商 / 提示词 / 词库 / **账号** / **我的套餐** / **用户管理**。
  ⚠️ **tab 按权限显隐**：`canSeeTab()` —— 「LLM 供应商」「词库」「用户管理」只有 `user.role === 'admin'` 才渲染，其他 tab 谁都能看。
  「提示词」tab 里面再分两块：**系统提示词**（`PromptPanel.vue`，非管理员 `readonly` —— 能看不能改）和**我的指令**（`UserPromptsPanel.vue`）。`AccountPanel.vue` 管账号信息和改密码，`PlanPanel.vue` 管自己的套餐和申请，`AdminPanel.vue` 管用户 / 审核 / 邀请码池。
  当前 tab 记在 URL 的 `?tab=` 上（`selectTab()` 同步 query，watch query 反向切 tab）—— 刷新不会跳回第一个 tab。

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
- **改密码**（`POST /api/auth/change-password`）：必须填**当前密码**（错 → 400）；改完用 `deleteOtherSessions()` 把**其它设备上的会话作废**、当前浏览器的留着。挂 `requireLogin`（不查配额）—— 超配额也得能改密码。

## 权限分档（新增）

三档，**写全局配置的接口一律 `requireAdmin`**（不给本地工具放行）：

| 档 | 能干什么 | 守卫 |
|---|---|---|
| 游客 | 查词、读配置（provider 列表、系统提示词只读） | 无 |
| 登录用户 | 用 AI 能力（查配额）；改自己的自定义指令；查词 | `requireUser` / `requireLogin` |
| 管理员 | 供应商增删改 / 切生效 / 语气表 / 系统提示词保存·回滚·复制 / 重新加载词库 | `requireAdmin` |

- `routes/settings.ts` 的**写**接口和 `routes/dictionary.ts` 的 `reload` 都挂了 `admin` 守卫。
  守卫是**可选注入**的（`userStore` 不传 = 不鉴权）—— 单元测试靠这个绕开登录（`tests/settingsRoutes.test.ts`）。
- `routes/customPrompts.ts`（新增）：用户自定义指令的 CRUD + `run`。
  指令存在 `user_prompts` 表（`user_id` + `provider_id`）—— 换 provider 就是另一套。
  付费判定用 `plans.ts` 的 `isPaidPlan()`（免费版不能新建；管理员不限）。
- ⚠️ **`currentUser()` 以前只带 `{id, username}`** —— 现在带 `role` / `planId` 了（`services/requestContext.ts` 的 `AuthUser`），
  权限判断要用的就是这两个。

## 用户自定义指令（新增）

- **系统提示词**（管理员维护，全局）vs **用户指令**（用户自己写，每个 provider 一套）—— 两回事。
- 用户指令只是一段**自然语言**（比如「改成学术腔」）；跑起来时被塞进**系统提示词的 `rewriteInstruction` 能力**里
  → 所以 system 那一侧始终是管理员定的，用户碰不到。
- `run` 接口的指令文本**从库里取**，客户端只送 `{text, context?, sourceSentence?, extraInstruction?}`。
- `extraInstruction` = 结果卡片里「继续调整」补的那句，拼在用户指令后面。
- `rewriteInstruction` 能力新增了两个模板变量 `sourceSentence` / `hasSource` —— **默认模板不用它们**，
  渲染结果逐字节不变（`tests/promptParity.test.ts` 守着这条）。

## 邮箱注册与验证码（新增）

- **新用户必须用邮箱注册**：`POST /api/auth/register` 要 `username + email + password + code`，码是从邮箱发来的 6 位验证码（先调 `POST /api/auth/register/code`）。**老用户没邮箱也不受影响** —— 照常登录，改密码走「当前密码」那条路。
- **验证码表 `email_codes`**（邮箱 / 码 / 用途 `register|reset` / 试错次数 / 过期时间 / 用掉时间）：10 分钟有效、同一邮箱同一用途只留**最新一个**有效码（旧码发新的那一刻就作废）、最多试 5 次、60 秒冷却。
  关键方法在 `config/userStore.ts`：`createEmailCode()` / `lastEmailCodeAt()`（冷却判断）/ `consumeEmailCode()`（校验 + 作废）。
- **发信**（`services/mailer.ts`）：直接 `fetch` Resend 的 HTTP 接口 —— **不引 resend SDK**，server 侧保持零第三方依赖。`.env` 里 `RESEND_API_KEY` + `RESEND_FROM`（`翻译工作台 <noreply@ihuhao.com>`）。
  ⚠️ 发信域名要在 Resend 后台验证过并配好 DNS，否则 Resend 返回 `domain is not verified`，代码会变成 500「邮件发送失败：…」。
- **改密码两条路**：① `change-password`（要当前密码，只踢其它设备）② `password-reset`（邮箱验证码，不用旧密码，**踢光所有会话**）。
- 登录可以填**用户名或邮箱**（`authenticate()` 两边都查）。管理员能在「用户管理」里给用户**补邮箱**（`POST /api/admin/users/:id/email`）—— 老用户补了才能用邮箱重置密码。
- **消耗 token 的接口必须登录**（`routes/auth.ts` 的 `requireUser`）：`/api/translate`、`/api/rewrite-*`、`/api/providers/:id/test`、`.../dry-run`。
- **本地工具免登录**：浏览器经 Vite 代理过来时后端看到的 IP 也是 127.0.0.1，所以**不能只看 IP** —— 靠浏览器特有的 `Origin` / `Sec-Fetch-Site` 头区分。回环 + 无这两个头 = 本地工具（curl / CLI）→ 放行但不记账。`.env` 的 `AUTH_LOCAL_BYPASS=0` 可关。
- **用量记账**：`ChatResult.usage` 由三个客户端填充（Ollama `prompt_eval_count`/`eval_count`；OpenAI `prompt_tokens`/`completion_tokens`；Anthropic `input_tokens`/`output_tokens` + cache 两段）→ service 层在 `chat()` 成功后、`extractOutput()` **之前**记 → 落到 `token_usage` 表（每人每天一行，UPSERT 累加）。
- **「记到谁头上」用 AsyncLocalStorage**（`services/requestContext.ts`）：登录守卫 `runAsUser(user, next)` 把身份挂进请求上下文 —— 并发安全，且**不用改任何 service 的方法签名**。
- **`day` 列是 INTEGER（YYYYMMDD）**，不是 TEXT —— SQLite 没有真正的 DATE 类型，INTEGER 比较没有 affinity 歧义。日/月边界按**服务端本地时区**算。
- CLI 不经过 HTTP → 不计入任何用户。

## 套餐、配额与管理员（新增）

- **套餐定义在代码里**（`server/src/config/plans.ts`），库里只存 `users.plan_id`。四档：free（2万/日、10万/月，新用户默认）/ basic / pro / unlimited（0/0 = 不限，只能管理员发码）。**配额单位是 token**（输入+输出）。
- **超额 = 直接拒绝**：`requireUser` 守卫放行前查「今日/本月已用 vs 上限」，超了 `403` + 文案。**管理员不受限**。只挡消耗 token 的那批接口 —— 查词/设置/**套餐申请**不受影响（超配额也得能点升级）。
- **三条守卫**（都在 `routes/auth.ts`）：`requireUser`（登录 + 查配额）、`requireLogin`（只要登录，不查配额，给套餐路由用）、`requireAdmin`（登录 + role=admin）。后两个**不给本地工具放行**。
- **两条升级路**：① 邀请码（一次性永久升级，抢码用 `UPDATE ... WHERE used_by IS NULL AND revoked_at IS NULL`，并发下只成功一个）② 升级申请（管理员审核通过时顺手改 `users.plan_id`）。
- **角色 / 禁用**：`users.role`（user/admin）+ `users.disabled_at`。`.env` 的 `ADMIN_USERNAME` + `ADMIN_USERS`（逗号分隔）是管理员名单，**每次启动都把名单里的人提成 admin**。禁用时**删掉他的所有会话**（立即生效）。
- **老库迁移**：`config/db.ts` 的 `ensureColumns()` —— `ALTER TABLE ADD COLUMN` 没有 `IF NOT EXISTS`，重复跑会报 duplicate column，所以先查 `PRAGMA table_info` 再决定加不加。

## 调试日志

`server/.env` 的 `DEBUG=1` 一个开关同时控制 server 端（`utils/debug.ts` 的 `debugLog`）和浏览器端（前端通过 `GET /api/debug` 拿到开关后给 `window.fetch` 打补丁）。**前端不单独配 .env。**

## 配置

- `server/.env`（**不进 git**）：`PORT` / `OLLAMA_HOST` / `OLLAMA_MODEL` / `MAX_CANDIDATES` / `<PROVIDER_ID>_KEY` / `VOCAB_PATH` / `DB_PATH` / `ADMIN_USERNAME` / `ADMIN_PASSWORD` / `ADMIN_USERS` / `AUTH_LOCAL_BYPASS` / `RESEND_API_KEY` / `RESEND_FROM` / `DEBUG`。启动时自动加载（`config/env.ts`），`SERVER_ENV_FILE` 可换文件。
- `server/data/app.db`（**不进 git**）：配置 + 用户 + 会话 + token 用量 + 升级申请 + 邀请码池。库里**永远没有明文 key** —— provider 的 `apiKeyEnv` 只记变量名。
- 切换生效 provider **即时生效**，不用重启后端（服务层每次请求都重新问 `activeProvider.ts` 的 getter）。
- 改 `VOCAB_PATH` 后可在设置页「词库」tab 点「重新加载」，不用重启。

## 测试约定

后端用 `vitest` + `supertest`，mock 从 `tests/helpers.ts` 注入。

**手测后端别用真实库** —— 看 `docs/临时库测试指南.md`：临时 env + 临时 SQLite + 另一个端口的后台 server，
测完 `rm -rf` 一下就干净了；里面还记了几个真踩过的坑（冷却测不出来、SQL 漏一列 → 静默 null 等）。

**`tests/promptParity.test.ts` 是迁移安全网**：默认提示词集的渲染结果必须和迁移前的 builder 逐字节相同。它的期望值**不能手抄** —— 期望值取自 `tests/legacy/prompts/` 里冻结的原始实现（**不要改那个目录**）。手抄的期望值会和模板一起错，测试反而是绿的。

其余测试文件覆盖见 `server/README.md` 的测试表格。

## 容易踩的坑

- **`requireUser` 会查配额** —— 新加「消耗 token 的接口」要挂它的时候想清楚：超配额的用户会被 403 挡掉，这是有意的。但套餐申请 / 兑换要挂 **`requireLogin`**（不查配额），否则超配额的用户连升级都点不了。
- **`requireLogin` / `requireAdmin` 不给本地工具（curl / CLI）放行** —— 只有 `requireUser` 有那个 bypass。管理操作必须有真人身份。
- **守卫不把用户塞进 `req`** —— 拿「当前登录用户」用 `currentUser()`（`services/requestContext.ts`，`requireLogin` / `requireAdmin` 挂上去之后身份在 AsyncLocalStorage 里）。别自己再去读 cookie 解析一遍。
- 后端库里 provider 的 `apiKeyEnv` 存的是**变量名**；`PUT /api/providers` 收 `apiKey` 会直接 400。
- `structuredOutput` 的 `temperature` 对 Anthropic 无效（API 已移除采样参数）；Anthropic 的 `max_tokens` 是必填，且是「思考 + 回复」的共同上限（默认 16000）。
- OpenAI 的 `strict: true` 只支持 JSON Schema 子集 → 代码里固定 `strict: false`。
- 前端浮动菜单的二级面板在**菜单内部展开**，不是飞出式子菜单（菜单 `overflow: hidden`，飞出式会被裁掉）。
- 后端 `src/` 下所有 import 都带 `.js` 后缀（ESM + tsx）。
- **`node:sqlite` 必须用 `createRequire` 加载**（`config/db.ts`）：Vite 5 的内置模块列表里还没有 `sqlite`，直接 `import from 'node:sqlite'` 会被打包器当成 npm 包去解析（`Failed to load url sqlite`）—— vitest 和前端构建都会炸。
- 库开了 WAL，会生成 `app.db-wal` / `app.db-shm` 两个附属文件 —— `server/.gitignore` 里是 `data/*.db*`。
- 后端启动日志**只打印账号名，不回显密码**（密码在 `server/.env` 的 `ADMIN_PASSWORD` 里）。
