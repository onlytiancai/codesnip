# tiptap-vue3-demo

Tiptap 富文本编辑器 demo：中英双向逐句翻译 + 英文写作 AI 助手（选区上下文菜单 / 语气改写 / 语法检查）+ 本地词库查词。前端 Vue 3 + Vite + vue-router，后端 Express（独立项目，见 `server/`）。

**LLM 后端可换**：本地 Ollama、任何 OpenAI 兼容端点、Anthropic Claude —— 在 `/settings` 子页面里配置。**提示词全部在 JSON 里**，可以在同一个页面改，改动留历史、能回滚；每个 provider 有自己独立的一套提示词。

## 启动

```bash
pnpm install
pnpm dev      # 启动 http://127.0.0.1:5173
```

后端要单独跑（提供 `/api/translate`、`/api/rewrite-*`、`/api/grammar-check`、`/api/dictionary/*`、`/api/auth/*`）：

```bash
cd server
pnpm install
pnpm dev      # 启动 http://127.0.0.1:8787
```

> Vite 已配置 `/api` 代理到后端，前端代码里直接 `fetch('/api/...')` 即可。

后端第一次启动会在 `server/data/app.db` 建库（SQLite），并**自动导入**老的 `server/data/config.json` + `history.json`（provider、提示词、语气表、版本历史）。到 **http://127.0.0.1:5173/settings** 可以配 LLM provider、改提示词、看历史、回滚、试运行、看词库状态。

## 登录

后端现在有用户体系：**所有消耗 token 的接口都必须登录**（翻译、改写、语法检查、试运行、provider 探活）。

- 打开 http://127.0.0.1:5173 会自动跳到 **/login**。
- 第一次启动后端时会自动建一个账号：**用户名 `admin`，密码见 `server/.env` 的 `ADMIN_PASSWORD`（默认 `admin123`）**。后端日志里只打印用户名，不回显密码。
- 登录页上可以切「注册」，自己再开一个账号。
- 登录态是 httpOnly cookie（30 天），刷新浏览器不用重新登录。

**侧栏底部**（设置项上方）显示当前用户、套餐和 token 消耗：

```
👤 alice          免费版
▓▓▓░░░░░░░  今日 12,340 / 20,000
今日  入 1,234 · 出 567
本月  入 12,345 · 出 6,789
```

数字每 20 秒刷新一次，翻译完很快能看到涨。配额条超了会变红。

### 套餐与配额

新注册的用户是**免费版**（每日 2 万 token / 每月 10 万 token）。配额按 **token**（输入+输出）算，
用满了 AI 接口会返回 `403`「免费版今日配额已用完（20,000 token），请升级套餐」——查词、设置、申请升级都不受影响。

升级有两条路（在设置页「**我的套餐**」里）：

- **邀请码** —— 管理员发的，输入就**立刻升级**（一次性、永久）。
- **提交申请** —— 管理员审核通过后生效。

管理员（`server/.env` 的 `ADMIN_USERNAME` / `ADMIN_USERS`）在设置页「**用户管理**」里能：
看所有用户的今日/本月用量、禁用/启用用户、直接改套餐、审核升级申请、批量生成邀请码。
详见 [server/README.md](server/README.md#套餐配额与管理员)。

### 本地工具免登录

`curl` / CLI 这类请求（来自 127.0.0.1、且不带 `Origin` / `Sec-Fetch-Site` 头）会**免登录放行**，所以 `server/curl-tests.sh` 不用改。但这类请求没有用户身份，**不计入任何人的 token 用量**。

浏览器请求一定会带 `Sec-Fetch-Site`（POST 还会带 `Origin`），所以浏览器必须登录。

想关掉这个放行：`server/.env` 里 `AUTH_LOCAL_BYPASS=0`。

### 调试日志

`server/.env` 里 `DEBUG=1` 打开调试输出（后端日志 + 浏览器 console.log 共用一个开关）：

- **server 端**：启动信息、登录/登出/401、本地放行、每次 LLM 调用的能力/模型/token/耗时、token 落库、配置变更。
- **浏览器 console**：`[debug] GET /api/... → 200 (12ms)` 这样的每行请求日志，加上登录状态变化。

关掉就把 `DEBUG` 那行删掉或置 0。

命令行工具也走同一份配置：

```bash
cd server
pnpm cli --help
pnpm cli translate --text "教育对孩子很重要。"
pnpm cli translate --to zh --text "Education is very important for children."   # 英译中
```

手工调 prompt / 模型又不想起 server 的话，后端带了个 CLI：

```bash
cd server
pnpm cli --help
pnpm cli selection --text "very important" --kind phrase --context "Education is very important for children."
pnpm cli sentence --text "Education is very important for children." --operation rewrite --tone formal
```

细节见 [server/README.md](server/README.md#手工测试-cli)。

## 主要功能

- **可收缩的左侧栏**：菜单项切换翻译方向（**中译英** / **英译中**），「⚙️ 设置」钉在左下角。收缩状态存 localStorage。侧栏在所有路由常驻。
- **登录 + 用户信息**：消耗 token 的功能都要登录（见下面的「登录」一节）。侧栏设置项上方显示当前用户名 + 今日/本月 token 消耗（输入 / 输出分开）。
- **左右两栏**（宽屏），左边是源文、右边是译文 —— 方向随侧栏菜单切
- **两个方向互不继承**：中译英和英译中是**两套独立状态** —— 在中译英的中文框输入的字，切到英译中不会带过去（英译中的中文框是空的），英文框同理；每个方向各自存着内容，切回来还在
- **逐句翻译**：在源文框写内容 → 选语气（可选）→ 点"翻译 →" → 后端按句并发（限 3）调 `/api/translate` → 结果写到译文框。切句时**点不是见到就断**：`PHP8.4`、`3.14`、`www.a.com`、`Mr. Smith`、`e.g.`、`U.S.` 里的点都不算句号
- **实时翻译**（可选开关，默认关，中译英方向）：打开后中文改完**不用点翻译**，停手 0.8 秒就自动翻——**以句子为单位**，没改过的句子直接复用缓存不重复请求；写回时只替换改动的那一段，译文框不闪、滚动不跳
- **源侧清空 → 译侧跟着清**：中文清空后英文自动清空（译文已经没有意义了）
- **翻译语气**：源栏头部常驻下拉，8 种语气（中性 / 正式 / 口语 / 专业 / 友好 / 简洁 / 详细 / 有说服力）；选"默认语气"则请求里不带 `tone`，行为与早期版本完全一致
- **英译中只要翻译**：英译中模式下英文框是**源文**，没有中文原文可消歧，所以那套改写/润色/语法 AI 能力全部关掉 —— 菜单里只剩「📖 查词」
- **查词**：在英文框里选中**一个单词** → 菜单里点「📖 查词」→ 查本地词库（音标 / 词性释义 / 例句 / 短语 / 同反义词 / 记忆法）。纯本地，不经过模型
- **清空 / 复制跟着源译侧走**：工具栏右侧那个按钮是**源侧给「清空」、译侧给「复制」**，切方向时两个框的按钮自动互换 —— 中译英时中文框是「清空」、英文框是「复制」，英译中反过来
- **选区上下文菜单**：在英文框里选中内容，菜单**自动弹出**并贴着选区定位；菜单项随选区类型变化（详见下表）
- **点词即候选**：中译英的英文框里，**光标停在某个词的最左侧**（词在光标右边）或**鼠标点在这个词上**（不用先拖选），直接弹出这个词的**可替换候选词**面板，点一下候选词就替换掉原词（保留原大小写）。弹出的悬浮框顶栏是两个 tab：**其它候选单词** / **查词**。英译中模式不开候选词（那边只留查词）
- **左右光标联动**：在源文框点击 → 译文框对应句高亮；反之亦然。仅在另一侧获焦时才显示


## 选区上下文菜单

英文框里的 AI 能力有两个入口，都落到同一套菜单 / 结果卡片上：

1. **跟随选区的浮动菜单** —— 选中内容时自动弹出
2. **工具栏 AI 下拉** —— 点「✨ AI」展开：🎨 换个语气 / ✏️ 润色 / ✂️ 精简 / ➕ 扩写 / 🔍 检查语法 / 💬 自定义指令，作用于**光标所在句**，不需要选中任何字符

浮动菜单用 `editor.view.coordsAtPos()` 拿选区上下两条 client rect，`Teleport` 到 body 后按 `position: fixed` 定位。**定位按菜单的实测高度算**（`ResizeObserver` 盯着面板高度）——查词词条、语法结果卡片都很长，用估计值摆会掉到视口外被裁掉；上方空间不足时翻到选区下方，再夹进视口。整块菜单在 `mousedown` 上 `preventDefault`——否则 ProseMirror 会先吃掉选区，点候选时 range 已经没了。

菜单内容随选区类型变化（`src/composables/useSelectionKind.ts` 判定）：

| 选区 | 判定 | 菜单内容 |
|---|---|---|
| 只有光标，没有选中任何字符 | `from === to` | **不弹**（但工具栏按钮仍可用）；中译英下光标正好停在**词的最左侧**、或鼠标点在这个词上时例外 —— 直接弹这个词的候选词 |
| 选区跨句 / 跨段 | 落不进任何单一句子区间 | **不弹** |
| 单个单词 | `/^\w+$/` | 顶栏两个 **tab**：其它候选单词 / 查词 |
| 多词但未跨句 | 其它 | 顶栏两个 **tab**：其它候选单词 / 查词 |
| 正好一整句 | 选区 trim 后 == 整句 trim 后 | 改写语气 / 润色 / 精简 / 扩写 |

选中词 / 短语时，悬浮框顶栏是**两个 tab**：**其它候选单词**（默认落在这个 tab，直接给候选词）和 **查词**（本地词库）。语法检查、自定义指令这两项从悬浮框里去掉了 —— 它们在工具栏「✨ AI」下拉里都有（作用于光标所在句）。

英译中模式下 `allowAi=false`：顶栏 tab 不出现（候选词要调模型），菜单里只剩「📖 查词」；选中的不是单个单词时**根本不弹菜单**（菜单里空荡荡的）。

- **其它候选单词** → `/api/rewrite-selection`，候选可以是多词短语（如 `very important` → `essential` / `of great importance`）。prompt 明确要求**搭配正确、拼接后语法正确、与前后文连贯**。点候选即替换，单词/短语都保留原大小写（`matchCase`）。请求**防抖 250ms** —— 拖选时选区一直在变，不防抖的话每挪一下就发一次请求。
- **改写语气** → 展开 8 种语气（在菜单面板内，不是飞出式二级菜单——菜单有 `overflow: hidden`，飞出式的子菜单会被裁掉），等同于 `rewrite` + `tone`
- **润色 / 精简 / 扩写** → `improve` / `shorten` / `expand`
- **检查语法**（在工具栏「✨ AI」下拉里）→ `/api/grammar-check`，**只提示不自动改**。后端把模型写的文字**原样透传**（`{text}`），面板直接原样显示（`<pre class="ai-menu__grammar">`），不做任何解析——本地默认模型 `hy-mt2` 只有 1.8B，让它吐结构化 JSON 不可靠，解析只会把「模型回了段自然语言」变成 `cannot parse JSON`。所以连输出格式也不要：user message 就是英文原句本身，system prompt 让模型用中文写一两句大白话（指出哪错了、改成什么），没错误时回「这句话没有语法或标点错误。」
- **查词**（顶栏第二个 tab）→ `/api/dictionary/lookup`，**纯本地**（不调模型，零 API 开销）。显示音标、词性释义、例句、短语、同近义词、反义词、记忆法。词库只收词头（`abandon`），所以查变形（`abandons`）会退一步还原，命中时提示「词库里收的是『abandon』」
- **查不到也能继续查**：词库里没有这个词时（比如 `scenarios`），面板会列出**词库里和它相近的词** —— 一档是**公共前缀相关**的（scene / scenery），一档是**编辑距离最短的 5 个**（拼错的词靠它，recieve → receive）。点一下就直接看那个词的解释，不用关掉菜单重新选中
- **自定义指令…**（在工具栏「✨ AI」下拉里）→ 文本框 + 5 个常用指令快捷 chip（`Make it more academic.` 等），结果以**结果卡片**呈现：`✓ 接受` / `✕ 拒绝`，另有「继续调整」——在已有结果上再追加一句要求跑下一轮。接受 = `insertContentAt({from,to}, text)`，Ctrl+Z 可撤销。

Escape 逐层退出：面板 → 菜单 → 关闭菜单（tab 模式下没有「上一层」，Esc 直接关）。菜单关闭时会 `abort()` 掉在飞的请求。改写类请求都带上当前英文句对应的中文原文作 `sourceSentence`，提高消歧精度。

工具栏按钮的实现方式是给菜单加一个 `autoAction` prop：点击后菜单贴着光标弹出，并直接落到对应面板（`tone` → 语气列表，`improve` → 润色中…）。光标落在句间空白时归到前一句；光标不在任何段落里时按钮禁用。

## 编辑器内高亮

英文框里直接用颜色标出光标上下文，不用再开调试面板：

| 颜色 | 含义 |
| --- | --- |
| 淡绿（整段底色） | 光标所在的**段落** |
| 淡蓝 | 光标所在的**句子** |
| 黄色 | 光标所在的**单词** |
| 紫色 | **跨栏对齐**：对侧光标所在句对应的这一句 |

规则：

- 焦点在**英文框** → 画淡绿 / 淡蓝 / 黄三层。
- 焦点在**中文框** → 英文框画紫色的跨栏高亮 + 它所在段落的淡绿底
  （不画蓝/黄，因为中英之间没有词级对应）；中文框画紫色的跨栏高亮。
- 也就是说：**哪边没焦点，哪边就高亮「对侧光标对应的那一句」**；焦点所在的那边画自己的三层上下文。
  中文框不画淡绿/淡蓝/黄三层。

实现是 ProseMirror 的 `Decoration`（段落用 `Decoration.node`，句子/单词用 `Decoration.inline`），
**不是真的 `<mark>`** —— 编辑器的 HTML 就是翻译接口的入参，塞真 mark 会污染发给后端的文本和句子对齐。

浏览器 DevTools Console 里 `[debug] ...` 日志会打印对齐表 + 当前光标的双侧索引，方便排查错位。

## 工具栏

两侧 editor 工具栏都有：**B / I / S / U + Undo / Redo**，历史组右边再跟一个按钮，**跟着「谁是源侧」走**：

| | 源侧（源文） | 译侧（译文） |
|---|---|---|
| 中译英 | 中文框 → **清空** | 英文框 → **复制** |
| 英译中 | 英文框 → **清空** | 中文框 → **复制** |

「复制」把框里的内容复制成纯文本（段落和硬换行都还原成换行），复制后按钮变绿提示一下。「清空」清空源文 —— 译侧会跟着一起清（译文已经没有意义了）。

切方向时按钮自动互换，不用手动换；两个框的**内容**不跟着换 —— 每个方向各存各的（见「主要功能」里那条「两个方向互不继承」）。

英文框最右边只有一个 **✨ AI** 按钮，点开才是完整的指令清单（🎨 换个语气 / ✏️ 润色 / ✂️ 精简 / ➕ 扩写 / 🔍 检查语法 / 💬 自定义指令），都作用于**光标所在句**、不需要选中字符。清单默认收着 —— 六个按钮摊在工具栏上会把半栏宽度占满。

下拉菜单跟浮动菜单一样 **Teleport 到 body**（编辑器外框是 `overflow: hidden`，直接放在工具栏里会被裁掉），用 `position: fixed` 定位、右对齐按钮；下方放不下就翻到按钮上面。点外面、按 Esc、选一项都会收起。

## 设置页（/settings）

侧栏左下角「⚙️ 设置」进这个子页面（独立路由 `/settings`，不是弹层 —— 提示词编辑要占的地方不小）。五个 tab：
**LLM 供应商 / 提示词 / 词库 / 我的套餐 / 用户管理**（最后一个只有管理员能看见）。

### LLM 供应商

每个 provider 一张卡片，可以**新增 / 编辑 / 删除 / 设为生效 / 测试连接 / 拉模型列表**。字段：

| 字段 | 说明 |
|---|---|
| 类型 | Ollama（本地）/ OpenAI 兼容（含 DeepSeek、Moonshot、vLLM…）/ Anthropic |
| Base URL | Ollama 填 host，其余填 baseURL |
| 模型 | 可以从「拉取模型列表」里点一下自动填 |
| 环境变量名 | key 在 `server/.env` 里，这里只填变量名（如 `MINIMAX_ANTHROPIC_KEY`）；留空 = 不用 key |
| 结构化输出 | 强制 JSON Schema / 只强制 JSON 对象 / 只靠提示词 / 不要求 JSON |
| max tokens、token 参数名、温度、超时 | 见 `server/README.md` 的 provider 配置表 |

切换生效的 provider **即时生效，不用重启后端**。

> **API Key 不再由这个页面保存** —— 它在 `server/.env` 里（不进 git），页面只记「读哪个变量名」。

### 提示词

顶部选 provider（每个 provider 的提示词**完全独立**），左边选能力（翻译 / **英译中** / 选区候选 / 整句改写 / 语法检查 / 自定义指令 / 语气表），右边编辑：

- System 模板、User 模板（两个 textarea）
- 可用变量 chip，点一下就插进 User 模板
- 输出模式：纯文本 / 格式化 JSON（JSON 模式可编辑 JSON Schema）
- **保存** = 追加一个版本（可填备注）
- **版本历史**：每条能「查看」或「回滚」——回滚是把旧内容复制成一个**新版本**（历史只增不减）
- **试运行**：填示例输入 → 用**正在编辑的这个 provider** 真跑一次 → 显示实际发出的 system / user、模型原始输出、解析结果（会真的消耗 API 额度）

也可以「从别的 provider 复制」——把调好的 Ollama 提示词搬给 Claude 再改。

### 词库

查词用的本地词库。**路径在 `server/.env` 的 `VOCAB_PATH` 里改**（JSONL，一行一个词条），
这个 tab 只读展示：

- 当前路径 + 「已加载 N 个词」
- **重新加载** —— 后端重读 `.env` 并重建索引，改完 `.env` 不用重启后端
- 测试查询框 —— 输个单词看命中不命中，用来验证路径改对了没

### 我的套餐

- **当前套餐卡**：套餐名 + 今日/本月配额条（超了变红 + 一行提示）
- **升级**：① 邀请码兑换（立刻生效）② 提交申请（管理员审核后生效），可带备注
- **我的申请**：状态（待审核 / 已通过 / 已驳回）+ 审核意见

### 用户管理（仅管理员）

- **用户表**：用户名 / 套餐（可直接改）/ 今日 / 本月 token / 状态（正常 / 已禁用），
  「用量」按钮展开**近 30 天**每日明细，「禁用 / 启用」即时生效（禁用会踢掉他的会话）
- **升级审核**：待审列表 → 通过 / 驳回（通过后立刻改套餐）
- **邀请码池**：选套餐 + 数量 + 批次名 → **批量生成**（生成的码可一键复制），
  下面列出全部码的状态（未使用 / 已使用 / 已作废）和使用者，未用的能作废

## 目录

```
src/
├── Root.vue                             # 根组件，只有 <router-view />
├── App.vue                              # 应用外壳：AppSidebar + <RouterView />
├── router.ts                            # 路由：/login（顶层）+ 外壳 / → 编辑器、/settings + 登录守卫
├── main.ts                              # 入口（顺便问后端要 DEBUG 开关）
├── style.css                            # 全局 + .ProseMirror + .ai-menu* + 侧栏 + 编辑器内高亮 + 设置页 + 登录页样式
├── pages/
│   ├── EditorPage.vue                   # 编辑器页：源/译两栏（方向随侧栏切）+ 联动
│   ├── SettingsPage.vue                 # 设置页（路由 /settings）：供应商 / 提示词 / 词库 三个 tab
│   └── LoginPage.vue                    # 登录页（路由 /login，不套侧栏外壳）+ 注册切换
├── components/
│   ├── AppSidebar.vue                   # 左侧栏：方向切换 + 用户信息/token 用量 + 左下角「⚙️ 设置」
│   ├── ChineseEditor.vue                # Tiptap 中文编辑器（中文切句 + 清空/复制按钮 + setHtml）
│   ├── TiptapEditor.vue                 # Tiptap 英文编辑器 + 选区菜单锚点（aiEnabled 控制 AI 那套）
│   ├── AiContextMenu.vue                # 选区上下文菜单 / 面板 / 结果卡片 / 查词面板
│   ├── ProviderPanel.vue                # provider 列表 + 编辑表单（baseUrl / model / 环境变量名 / 结构化输出）
│   ├── PromptPanel.vue                  # 提示词编辑（按 provider）+ 版本历史 + 回滚 + 试运行
│   └── DictionaryPanel.vue              # 词库 tab：路径（只读）+ 重新加载 + 测试查询
├── composables/
│   ├── useWritingContext.ts             # 段落/句子/单词定位 + 文本偏移↔doc 位置换算（sentenceIndex 全局）
│   ├── useSelectionKind.ts              # 选区分类 word/phrase/sentence；getSentenceAtCursor 供工具栏用
│   ├── useCopyContent.ts                # 「复制」按钮共用逻辑（纯文本提取 + 剪贴板 + 已复制提示）
│   ├── useTones.ts                      # 语气表（label 从后端拉，拉不到用内置的）
│   ├── useMode.ts                       # 翻译方向（zh2en / en2zh），模块级单例 + localStorage
│   ├── useAuth.ts                       # 当前登录用户 + 今日/本月 token 用量，模块级单例
│   └── highlightContext.ts              # Decoration 工厂：跨栏高亮 + 段落/句子/单词三层
├── constants/
│   ├── tones.ts                         # 8 种语气的 id（前端唯一来源）+ 内置 label
│   └── aiActions.ts                     # 菜单项 / 工具栏按钮共用的动作枚举（含 lookup）
├── services/
│   ├── http.ts                          # 统一 fetch 封装（把后端 {error} 捞出来）
│   ├── rewrite.ts                       # fetch rewrite-selection / -sentence / grammar-check / -instruction
│   ├── translate.ts                     # fetch /api/translate（pLimit 并发 3，可带 tone + 方向）
│   ├── dictionary.ts                    # fetch /api/dictionary/{status,lookup,reload}
│   ├── settings.ts                      # fetch 设置相关端点（provider / 提示词 / 版本 / 试运行）
│   └── auth.ts                          # fetch /api/auth/{login,register,logout,me} + /api/debug
└── utils/
    ├── matchCase.ts                     # Good→Great / good→great / GOOD→GREAT
    └── debug.ts                         # 调试日志开关（跟后端共用 server/.env 的 DEBUG）
```

## 脚本

```bash
pnpm dev          # vite dev (port 5173)
pnpm build        # vue-tsc + vite build
pnpm preview      # 预览构建产物
pnpm type-check   # vue-tsc --noEmit
```

## 配置

- Vite 端口：见 `vite.config.ts`
- API 代理：`/api → http://127.0.0.1:8787`
- `.npmrc`：`auto-install-peers=true`（避免 pnpm peer 警告）
- **后端配置**：`server/.env`（端口 / Ollama 地址 / API Key / 词库路径 / 登录 / 调试开关），**不进 git**。
  后端启动时自动加载（`server/src/config/env.ts`），也可以 `SERVER_ENV_FILE` 指到别的文件。
- **LLM 配置**：provider + 每个 provider 的提示词现在存在 **SQLite**（`server/data/app.db`，不进 git）里，在 `/settings` 页里改。
  老版本用的 `server/data/config.json` 只作为**首次导入源**（自动导入后不再回写）。
  API Key 在 `server/.env` 里，设置页只填「环境变量名」。细节见 [server/README.md](server/README.md#配置)
