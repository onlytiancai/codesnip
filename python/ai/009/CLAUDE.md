# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

零构建、零 npm 的纯前端单页应用：**神经网络小课堂**（中英双语互动教程，10 章）。所有依赖走 CDN（Vue 3、vue-router、marked、DOMPurify、html2canvas、Chart.js、KaTeX），不打包、不编译。

## 启动开发服务器

**必须用 HTTP 服务器，不能用 `file://`**（浏览器会拒绝 `fetch('chapters.json')`）。

```bash
cd /Users/huhao/src/codesnip/python/ai/009
/Users/huhao/.pyenv/versions/3.11.9/bin/python3 -m http.server 8765
# 浏览器访问 http://localhost:8765
```

## 重新生成配图

```bash
bash scripts/gen_all.sh
# 或单章
/Users/huhao/.pyenv/versions/3.11.9/bin/python3 scripts/gen_ch03.py
```

依赖 `numpy` + `matplotlib`（见 `scripts/requirements.txt`）。`scripts/_fonts.py` 自动处理 CJK 字体回退（macOS → Windows → Linux）。

## 代码架构

### 入口与启动顺序（`index.html` → `app.js`）

1. `index.html` 通过 `<script type="importmap">` 把 `"vue"` 解析到 jsdelivr 的 ESM 版本
2. UMD 库（marked、DOMPurify、html2canvas）暴露在 `window.*`
3. `progress.config.js` 在 `app.js` 之前定义 `window.PROGRESS_CONFIG`（阈值 + localStorage 键名）
4. `assets/js/app.js` 的 `bootstrap()`：`fetch('chapters.json')` → 建 store → 建 router → 注册全局组件 → `app.mount('#app')`

### 状态管理（无 Pinia，用 reactive + provide/inject）

- `assets/js/store.js`：`reactive(state)` 单例 + `provide(STORE_KEY, ...)`。读用 `useStore()`（带 `readonly` 保护），写用 `actions.xxx`。
- 跨子应用：在子 Vue app 上调用 `provideStore(subApp)`（见 `ChapterView.mountBlocks`），同一 store 引用传递。
- 持久化：所有写操作都同步调 `progress.js` 里 `saveXxx()` 写 localStorage（键名见 `progress.config.js` 的 `STORAGE_KEYS`）。

### 路由（hash 模式）

`assets/js/router.js`：`/` → `HomeView`、`/ch/:id` → `ChapterView`、`/cert` → `CertView`。`HomeView`/`ChapterView`/`CertView` 通过 `router-view :key="$route.fullPath"` 强制重挂载，简化清理。

### 章节内容渲染管线（核心流程）

`ChapterView.loadChapter()`：
1. `fetch` 章节 markdown（zh/en 根据 `state.language`）
2. `markdown.js` 的 `renderMarkdown()`：
   - **Step 0**：等 `window.katex` 加载
   - **Step 1**：用正则提取 `::: type ... :::` 块（见下表），替换为占位 `<div data-block-type data-block-args data-block-body>`
   - **Step 2**：用 `extractMath()` 提取 `$$..$$` / `$..$`，KaTeX 预渲染 → 占位符 `@@MATH_BLOCK_N@@` / `@@MATH_INLINE_N@@`
   - **Step 3**：`marked.parse()`
   - **Step 4**：把 math 占位符替换回 KaTeX HTML
   - **Step 5**：`DOMPurify.sanitize()` 清洗（注意 `ADD_TAGS` / `ADD_ATTR` 白名单）
3. `v-html` 注入到 `#chapter-content`
4. `mountBlocks()` 扫描 `[data-block-type]` 占位 div，对每个块 `createApp({render: () => h(Comp, {data: parsed, chapterId})})` 子 app → 挂载（失败不影响其他块）
5. `renderMath()` 调 `window.renderMathInElement` 兜底（万一 extractMath 漏了）
6. `setupScrollTracking()` 监听 scroll，每 5% 写一次进度

### `::: type ... :::` 容器（`markdown.js`）

| 容器 | 解析器 | Vue 组件 |
|---|---|---|
| `::: quiz q-id single\|multiple\|short` | `parseQuizBlock()` | `Quiz.js` |
| `::: chart caption="..."` | `parseChartBlock()` | `ChartBlock.js` |
| `::: graph` | `parseGraphBlock()` | `ComputeGraph.js` |
| `::: network` | `parseNetworkBlock()` | `NetworkViz.js` |
| `::: train-demo :steps=200 :lr=0.5` | `parseTrainDemoBlock()` | `TrainDemo.js` |
| `::: formula` | `parseFormulaBlock()` | `Formula.js` |

新增容器类型需要：(1) `BLOCK_TYPES` 数组；(2) 解析器；(3) `COMP_MAP` 注册；(4) `app.js` 全局注册组件。

### 章节完成判定（`ChapterView.checkCompletion`）

- 答题正确率 ≥ 60%（**分母只算判了分的题**：单选/多选 `correct=true/false`；简答 `correct=null` 不计入，否则 4 道里 2 道简答会卡死 60% 阈值）
- 且（答完全部题 ∨ 滚动 ≥ 80%）
- 满足即 `actions.completeChapter(m.id)`，弹 toast
- 触发：`scroll` 事件 + `watch` 答题变化（不依赖滚动）

### 证书颁发（`chapters.json` 的 `certificate`）

`require_all_chapters: true` + `min_pass_rate: 0.6`。`CertView.js` 用 `html2canvas` 把证书 DOM 转 PNG 下载。

## 加新章节流程（参见 README.md "如何加新章"）

1. `content/` 下创建 `ch##_xxx_zh.md` 和 `ch##_xxx_en.md`（保持章节 id 前缀与 `chapters.json` 一致）
2. `chapters.json` 追加新章（注意 `prerequisites` 数组决定章节解锁）
3. `scripts/gen_ch##.py`（可选，配图脚本）
4. 重新加载浏览器即可

## 关键文件速查

| 路径 | 职责 |
|---|---|
| `index.html` | 唯一 HTML 入口；importmap + CDN + 4 个 CSS |
| `chapters.json` | 章节元数据 + 证书配置 |
| `progress.config.js` | 完成阈值 + 证书门槛 + localStorage 键名 + schema 版本号 |
| `assets/js/app.js` | 启动入口（bootstrap + KaTeX strict 补丁） |
| `assets/js/store.js` | 迷你 store（reactive + provide/inject） |
| `assets/js/router.js` | vue-router hash 配置 |
| `assets/js/markdown.js` | marked + `:::` 容器 + KaTeX 预渲染 + DOMPurify |
| `assets/js/progress.js` | localStorage 读写 + schema 迁移 + 汇总统计 |
| `assets/js/i18n.js` | UI 元素 zh/en 翻译表 |
| `assets/js/components/*.js` | 13 个 Vue 组件（`AppShell`/`HomeView`/`ChapterView`/`CertView` + 6 个容器组件 + 4 个工具组件） |
| `assets/css/{theme,app,quiz,cert}.css` | CSS 变量、布局、测试题、证书（含 `@media print`） |
| `content/ch##_*.md` | 章节 markdown 内容（zh/en） |
| `scripts/_fonts.py` | CJK 字体配置（其他 gen 脚本都先 import 它） |
| `scripts/gen_ch##.py` | 配图生成脚本（numpy + matplotlib） |
