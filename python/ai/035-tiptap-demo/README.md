# tiptap-vue3-demo

Tiptap 富文本编辑器 demo：中英逐句翻译 + 英文写作 AI 助手（选区上下文菜单 / 语气改写 / 语法检查）。前端 Vue 3 + Vite + vue-router，后端 Express（独立项目，见 `server/`）。

**LLM 后端可换**：本地 Ollama、任何 OpenAI 兼容端点、Anthropic Claude —— 在 `/settings` 子页面里配置。**提示词全部在 JSON 里**，可以在同一个页面改，改动留历史、能回滚；每个 provider 有自己独立的一套提示词。

## 启动

```bash
pnpm install
pnpm dev      # 启动 http://127.0.0.1:5173
```

后端要单独跑（提供 `/api/translate`、`/api/rewrite-*`、`/api/grammar-check`）：

```bash
cd server
pnpm install
pnpm dev      # 启动 http://127.0.0.1:8787
```

> Vite 已配置 `/api` 代理到后端，前端代码里直接 `fetch('/api/...')` 即可。

后端第一次启动会在 `server/data/` 下生成 `config.json`（provider + 提示词）和 `history.json`（版本历史）。到 **http://127.0.0.1:5173/settings** 可以配 LLM provider、改提示词、看历史、回滚、试运行。

命令行工具也走同一份配置：

```bash
cd server
pnpm cli --help
pnpm cli translate --text "教育对孩子很重要。"
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

- **左右两栏**（宽屏），中文编辑器 + 英文编辑器
- **逐句翻译**：在中文框写中文 → 选语气（可选）→ 点"翻译 →" → 后端按句并发（限 3）调 `/api/translate` → 结果写到英文框
- **翻译语气**：中文栏头部常驻下拉，8 种语气（中性 / 正式 / 口语 / 专业 / 友好 / 简洁 / 详细 / 有说服力）；选"默认语气"则请求里不带 `tone`，行为与早期版本完全一致
- **清空中文**：中文框工具栏右侧"清空"按钮，只清中文内容，英文译文保留
- **选区上下文菜单**：在英文框里选中内容，菜单**自动弹出**并贴着选区定位；菜单项随选区类型变化（详见下表）
- **左右光标联动**：在中文框点击 → 英文框对应英文句高亮；反之亦然。仅在另一侧获焦时才显示。

## 选区上下文菜单

英文框里的 AI 能力有两个入口，都落到同一套菜单 / 结果卡片上：

1. **跟随选区的浮动菜单** —— 选中内容时自动弹出
2. **工具栏常驻按钮** —— 🎨 改写语气 / ✏️ 润色 / ✂️ 精简 / ➕ 扩写 / 🔍 检查语法 / 💬 自定义指令，作用于**光标所在句**，不需要选中任何字符

浮动菜单用 `editor.view.coordsAtPos()` 拿选区上下两条 client rect，`Teleport` 到 body 后按 `position: fixed` 定位；上方空间不足时翻到选区下方，再夹进视口。整块菜单在 `mousedown` 上 `preventDefault`——否则 ProseMirror 会先吃掉选区，点候选时 range 已经没了。

菜单项随选区类型变化（`src/composables/useSelectionKind.ts` 判定）：

| 选区 | 判定 | 菜单项 |
|---|---|---|
| 只有光标，没有选中任何字符 | `from === to` | **不弹**（但工具栏按钮仍可用） |
| 选区跨句 / 跨段 | 落不进任何单一句子区间 | **不弹** |
| 单个单词 | `/^\w+$/` | 替换词 / 检查语法 / 自定义指令… |
| 多词但未跨句 | 其它 | 替换短语 / 自定义指令… |
| 正好一整句 | 选区 trim 后 == 整句 trim 后 | 改写语气 / 润色 / 精简 / 扩写 / 检查语法 / 自定义指令… |

- **改写语气** → 展开 8 种语气（在菜单面板内，不是飞出式二级菜单——菜单有 `overflow: hidden`，飞出式的子菜单会被裁掉），等同于 `rewrite` + `tone`
- **润色 / 精简 / 扩写** → `improve` / `shorten` / `expand`
- **替换词 / 替换短语** → `/api/rewrite-selection`，候选可以是多词短语（如 `very important` → `essential` / `of great importance`）。prompt 明确要求**搭配正确、拼接后语法正确、与前后文连贯**。点候选即替换，单词/短语都保留原大小写（`matchCase`）。
- **检查语法** → `/api/grammar-check`，**只提示不自动改**。后端把模型写的文字**原样透传**（`{text}`），面板直接原样显示（`<pre class="ai-menu__grammar">`），不做任何解析——本地默认模型 `hy-mt2` 只有 1.8B，让它吐结构化 JSON 不可靠，解析只会把「模型回了段自然语言」变成 `cannot parse JSON`。所以连输出格式也不要：user message 就是英文原句本身，system prompt 让模型用中文写一两句大白话（指出哪错了、改成什么），没错误时回「这句话没有语法或标点错误。」
- **自定义指令…** → 文本框 + 5 个常用指令快捷 chip（`Make it more academic.` 等），结果以**结果卡片**呈现：`✓ 接受` / `✕ 拒绝`，另有「继续调整」——在已有结果上再追加一句要求跑下一轮。接受 = `insertContentAt({from,to}, text)`，Ctrl+Z 可撤销。

Escape 逐层退出：面板 → 菜单 → 关闭菜单。菜单关闭时会 `abort()` 掉在飞的请求。改写类请求都带上当前英文句对应的中文原文作 `sourceSentence`，提高消歧精度。

工具栏按钮的实现方式是给菜单加一个 `autoAction` prop：点击后菜单贴着光标弹出，并直接落到对应面板（`tone` → 语气列表，`improve` → 润色中…）。光标落在句间空白时归到前一句；光标不在任何段落里时按钮禁用。

## 调试模式

顶栏右上角"显示调试 / 隐藏调试"按钮切换以下面板：
- 当前中文句子
- 当前段落 / 句子 / 单词
- 对应中文原文（传给改写 API）
- 右侧 meta "第 N 句 / 共 M 句"

浏览器 DevTools Console 里 `[debug] ...` 日志会打印对齐表 + 当前光标的双侧索引，方便排查错位。

## 工具栏

两侧 editor 工具栏都有：**B / I / S / U + Undo / Redo**。中文框在历史组右侧多一个 **清空** 按钮；英文框在历史组右侧多一排常驻 AI 按钮（🎨 语气 / ✏️ 润色 / ✂️ 精简 / ➕ 扩写 / 🔍 语法 / 💬 指令），作用于光标所在句、不需要选中字符。按钮标签压到 2 个字（完整文案在 `title` 上）——六个按钮挤在半栏宽度里，用全称会把工具栏撑到换行。

## 设置页（/settings）

编辑器顶栏右上角「⚙️ 设置」进这个子页面（独立路由 `/settings`，不是弹层 —— 提示词编辑要占的地方不小）。两个 tab：

### LLM 供应商

每个 provider 一张卡片，可以**新增 / 编辑 / 删除 / 设为生效 / 测试连接 / 拉模型列表**。字段：

| 字段 | 说明 |
|---|---|
| 类型 | Ollama（本地）/ OpenAI 兼容（含 DeepSeek、Moonshot、vLLM…）/ Anthropic |
| Base URL | Ollama 填 host，其余填 baseURL |
| 模型 | 可以从「拉取模型列表」里点一下自动填 |
| API Key | 存到 `server/data/secrets.json`（**不进 git**）；已存时留空表示不修改 |
| 结构化输出 | 强制 JSON Schema / 只强制 JSON 对象 / 只靠提示词 / 不要求 JSON |
| max tokens、token 参数名、温度、超时 | 见 `server/README.md` 的 provider 配置表 |

切换生效的 provider **即时生效，不用重启后端**。

### 提示词

顶部选 provider（每个 provider 的提示词**完全独立**），左边选能力（翻译 / 选区候选 / 整句改写 / 语法检查 / 自定义指令 / 语气表），右边编辑：

- System 模板、User 模板（两个 textarea）
- 可用变量 chip，点一下就插进 User 模板
- 输出模式：纯文本 / 格式化 JSON（JSON 模式可编辑 JSON Schema）
- **保存** = 追加一个版本（可填备注）
- **版本历史**：每条能「查看」或「回滚」——回滚是把旧内容复制成一个**新版本**（历史只增不减）
- **试运行**：填示例输入 → 用**正在编辑的这个 provider** 真跑一次 → 显示实际发出的 system / user、模型原始输出、解析结果（会真的消耗 API 额度）

也可以「从别的 provider 复制」——把调好的 Ollama 提示词搬给 Claude 再改。

## 目录

```
src/
├── Root.vue                             # 根组件，只有 <router-view />
├── App.vue                              # 编辑器页（路由 /）：顶栏 + 语气下拉 + 联动 + 调试开关
├── router.ts                            # 两条路由：/ 编辑器、/settings 设置
├── main.ts                              # 入口
├── style.css                            # 全局 + .ProseMirror + .ai-menu* + ctx 面板 + 设置页样式
├── pages/
│   └── SettingsPage.vue                 # 设置页（路由 /settings）：供应商 / 提示词 两个 tab
├── components/
│   ├── ChineseEditor.vue                # Tiptap 中文编辑器（中文切句 + 清空按钮）
│   ├── TiptapEditor.vue                 # Tiptap 英文编辑器 + 选区菜单锚点
│   ├── AiContextMenu.vue                # 选区上下文菜单 / 面板 / 结果卡片（autoAction 可直达某面板）
│   ├── ProviderPanel.vue                # provider 列表 + 编辑表单（baseUrl / model / key / 结构化输出）
│   └── PromptPanel.vue                  # 提示词编辑（按 provider）+ 版本历史 + 回滚 + 试运行
├── composables/
│   ├── useWritingContext.ts             # 段落/句子/单词提取（sentenceIndex 全局）
│   ├── useSelectionKind.ts              # 选区分类 word/phrase/sentence；getSentenceAtCursor 供工具栏用
│   ├── useTones.ts                      # 语气表（label 从后端拉，拉不到用内置的）
│   └── highlightSentence.ts             # Decoration 工厂：中英跨面板高亮
├── constants/
│   ├── tones.ts                         # 8 种语气的 id（前端唯一来源）+ 内置 label
│   └── aiActions.ts                     # 菜单项 / 工具栏按钮共用的动作枚举
├── services/
│   ├── http.ts                          # 统一 fetch 封装（把后端 {error} 捞出来）
│   ├── rewrite.ts                       # fetch rewrite-selection / -sentence / grammar-check / -instruction
│   ├── translate.ts                     # fetch /api/translate（pLimit 并发 3，可带 tone）
│   └── settings.ts                      # fetch 设置相关端点（provider / 提示词 / 版本 / 试运行）
└── utils/
    └── matchCase.ts                     # Good→Great / good→great / GOOD→GREAT
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
- **LLM 配置**：全部在 `server/data/config.json`（provider + 每个 provider 的提示词），在 `/settings` 页里改；API Key 在 `server/data/secrets.json`（不进 git）。细节见 [server/README.md](server/README.md#配置)
