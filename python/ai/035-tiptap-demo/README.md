# tiptap-vue3-demo

Tiptap 富文本编辑器 demo：中英逐句翻译 + 英文写作 AI 助手（选区上下文菜单 / 语气改写 / 语法检查）。前端 Vue 3 + Vite，后端 Express + Ollama（独立项目，见 `server/`）。

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

## 目录

```
src/
├── App.vue                              # 顶栏 + 语气下拉 + 联动 + 调试开关
├── main.ts                              # 入口
├── style.css                            # 全局 + .ProseMirror + .ai-menu* + ctx 面板样式
├── components/
│   ├── ChineseEditor.vue                # Tiptap 中文编辑器（中文切句 + 清空按钮）
│   ├── TiptapEditor.vue                 # Tiptap 英文编辑器 + 选区菜单锚点
│   └── AiContextMenu.vue                # 选区上下文菜单 / 面板 / 结果卡片（autoAction 可直达某面板）
├── composables/
│   ├── useWritingContext.ts             # 段落/句子/单词提取（sentenceIndex 全局）
│   ├── useSelectionKind.ts              # 选区分类 word/phrase/sentence；getSentenceAtCursor 供工具栏用
│   └── highlightSentence.ts             # Decoration 工厂：中英跨面板高亮
├── constants/
│   ├── tones.ts                         # 8 种语气的 id + 中文 label（前端唯一来源）
│   └── aiActions.ts                     # 菜单项 / 工具栏按钮共用的动作枚举
├── services/
│   ├── rewrite.ts                       # fetch rewrite-selection / -sentence / grammar-check / -instruction
│   └── translate.ts                     # fetch /api/translate（pLimit 并发 3，可带 tone）
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
