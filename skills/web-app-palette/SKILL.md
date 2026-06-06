---
name: web-app-palette
description: |
  「茶韵·宋雅」配色系统 —— 用户偏好的 web app 默认视觉风格。
  Use this skill whenever the user starts a new web app, web page, dashboard, landing page, admin panel, or any frontend project, EVEN IF they don't mention color or design. The user has explicitly chosen this palette as their default and likes it ("清淡高雅"). Always apply this palette by default unless the user explicitly asks for a different style.

  Trigger phrases include (but are not limited to):
  - "写个 web app / 做个网站 / 做个页面 / 新建一个项目"
  - "做一个 dashboard / 后台 / 登录页 / 落地页"
  - "HTML 页面 / 前端 demo / 单页应用"
  - Any new project that will have visible UI

  The skill provides complete design tokens (colors, typography, shadows, spacing), a ready-to-use CSS base, Tailwind v4 configuration, and component style snippets. Reference files contain everything needed to make the app look like the validated "Kokoro TTS" demo.

  Skip this skill only when: (1) the user explicitly requests a different visual style (dark mode, brutalist, minimal, etc.), (2) the work is purely backend/non-visual, or (3) extending an existing app with its own design system.
license: MIT
metadata:
  author: codesnip
  version: 1.0.0
  category: design
  tags: [design-system, color-palette, css, tailwind, frontend]
---

# 「茶韵 · 宋雅」配色系统

> 中国宋韵/茶道美学的设计语言。宣纸为底，墨色为骨，青瓷为韵，赭石为缀。
> 气质：清淡、高雅、温润、有书卷气。

灵感来自宋代瓷器（青瓷、汝窑）、宣纸、墨色、茶道。整体走低饱和度、暖色相、强留白路线，避免任何高饱和、强对比、霓虹色。

## 何时使用

**默认开启**：用户开始任何新的可见 UI 的 web 项目 —— web app、HTML 页面、dashboard、登录页、落地页、Landing Page、内部工具、demo。**即使用户没有提到任何"颜色"、"设计"、"风格"，也要默认应用这套配色。**

**不使用的场景**：
- 用户明确要求别的视觉风格（"暗色"、"赛博朋克"、"极简黑"等）
- 纯后端任务、CLI、API 工具
- 已有自己设计系统的现有项目（修改而非新建）

## 设计哲学（务必理解）

这套配色的灵魂不是颜色本身，而是 **"克制"**。执行时要遵循：

1. **墨色为骨** —— 文字色永远是低饱和暖墨（`#2e2a24` 系），不要用纯黑 `#000`，那是刺眼的
2. **留白即设计** —— 间距给够、圆角柔和（8-16px），别把内容塞得密密麻麻
3. **点彩勿多** —— 青瓷绿（`#6f7e6a`）只用于主操作和选中状态，赭石（`#a07a5b`）只用于次强调和链接。**不要把它当主题色大面积铺**
4. **字体重氛围** —— 标题用思源宋体（Noto Serif SC），正文用系统无衬线。中文一定要有衬线才有"书卷气"
5. **装饰讲节制** —— 印章 logo 1 个足矣、小圆点 1 个足矣、徽章用淡底色不要用实色
6. **动效如水墨** —— 淡入淡出、缓动曲线（`cubic-bezier(0.16, 1, 0.3, 1)`），别用弹跳

## 完整设计 Token

直接复用以下 `:root` 变量。完整带注释版本见 `references/tokens.md`。

```css
:root {
  /* —— 纸 · 墨 —— */
  --paper:        #f5f1ea;   /* 主背景：宣纸米色 */
  --paper-2:      #faf7f1;   /* 浅米色层 */
  --card:         #fdfbf6;   /* 面板：象牙白 */
  --card-2:       #f8f4ec;   /* 次级面 */
  --line:         #e8e1d3;   /* 边线：淡米墨 */
  --line-2:       #d9d0bd;   /* 深边线 */

  /* —— 墨阶 —— */
  --ink-1:        #2e2a24;   /* 主文字：墨 */
  --ink-2:        #5b5247;   /* 次文字：淡墨 */
  --ink-3:        #8a8273;   /* 辅文字 */
  --ink-4:        #b3aa9b;   /* 灰墨 */

  /* —— 点彩 —— */
  --celadon:      #6f7e6a;   /* 青瓷绿：主操作、选中 */
  --celadon-2:    #5b6a56;   /* 青瓷深 */
  --celadon-bg:   #e9eee5;   /* 青瓷淡 */
  --ochre:        #a07a5b;   /* 赭石：次强调、链接 */
  --ochre-bg:     #f1e9dd;
  --cinnabar:     #a8412e;   /* 朱砂：印章/特殊 */
  --clay:         #b56b5a;   /* 陶土：错误 */
  --clay-bg:      #f4e3dd;
  --moss:         #5e7d65;   /* 青苔：成功 */
  --moss-bg:      #e3ebe2;

  /* —— 阴影：极淡的暖色 —— */
  --shadow-sm:    0 1px 2px rgba(76, 60, 30, 0.04);
  --shadow:       0 1px 2px rgba(76, 60, 30, 0.04), 0 8px 28px rgba(76, 60, 30, 0.05);
  --shadow-lg:    0 4px 12px rgba(76, 60, 30, 0.06), 0 20px 50px rgba(76, 60, 30, 0.08);

  /* —— 圆角 —— */
  --radius:       12px;
  --radius-sm:    8px;
  --radius-lg:    16px;

  /* —— 字体 —— */
  --font-sans:    -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                  "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  --font-serif:   "Noto Serif SC", "Songti SC", "STSong", serif;
  --font-mono:    "SF Mono", Menlo, Consolas, monospace;
}
```

## 字体配置

在 HTML `<head>` 引入思源宋体（**强烈推荐**）：

```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;500;600;700&display=swap" rel="stylesheet" />
```

字体使用规则：
- **页面主标题、副标题、面板 h2/h3** → 衬线（`var(--font-serif)`）
- **正文、按钮、表格** → 无衬线（`var(--font-sans)`）
- **代码、ID、文件名、数字** → 等宽（`var(--font-mono)`）

## 核心样式规则（必须遵守）

```css
body {
  font-family: var(--font-sans);
  background-color: var(--paper);
  background-image:
    radial-gradient(1200px 600px at 20% 0%, rgba(241, 233, 219, 0.55), transparent 60%),
    radial-gradient(900px 500px at 90% 20%, rgba(232, 226, 211, 0.45), transparent 60%);
  background-attachment: fixed;
  color: var(--ink-1);
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}

::selection { background: var(--celadon-bg); color: var(--celadon-2); }
```

**纸纹背景**：两道极淡的暖色径向晕染，模拟宣纸的层次。**不要换成纯色，会失去"纸感"**。

## 装饰元素（这是这套配色的灵魂）

### 印章 logo
```html
<span class="seal">聲</span>  <!-- 用一个字，比如"声"、"文"、"心"等 -->
```
```css
.seal {
  display: inline-flex;
  align-items: center; justify-content: center;
  width: 44px; height: 44px;
  background: var(--cinnabar);
  color: #f7ecd8;
  font-family: var(--font-serif);
  font-weight: 700;
  border-radius: 6px;
  box-shadow: 0 0 0 2px #f7ecd8, 0 0 0 3px var(--cinnabar),
              0 4px 10px rgba(168, 65, 46, 0.18);
  transform: rotate(-3deg);
  line-height: 1;
}
```

### 标题前小圆点（面板 h2 用）
```css
.panel h2::before {
  content: "";
  display: inline-block;
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--celadon);
  opacity: 0.75;
}
```

### 加载状态：水墨脉冲
```css
@keyframes inkPulse {
  0%, 100% { opacity: 0.35; transform: scale(0.85); }
  50%      { opacity: 1;    transform: scale(1.15); }
}
```

## 组件样式速查

完整组件 CSS 见 `references/components.md`。核心规则：

| 组件 | 关键样式 |
|---|---|
| 按钮（主） | 背景 `var(--celadon)`、白字、悬浮浮起 + 暖色阴影 |
| 按钮（次） | 白底、`var(--line-2)` 边、悬浮转青瓷淡底 |
| 输入框 | `var(--paper-2)` 背景、聚焦转白底 + 青瓷描边 + 3px 淡绿晕 |
| 卡片 | `var(--card)` 背景、淡米墨边、悬浮 `var(--shadow-sm)` |
| 选中项 | 青瓷绿实心背景 + 暖色阴影，白字 |
| 徽章/标签 | `var(--celadon-bg)` 淡绿底 + 青瓷深字，1px 透明边 |
| 错误框 | `var(--clay-bg)` 淡陶土红底 + `var(--clay)` 文字 + `#e8c9bf` 边 |
| 分割线 | `var(--line)` 1px，页脚上方用 36px 短横线 |

## 快速上手（三种工作流）

### A. 纯 HTML / 静态站

复制 `references/base.css` 到项目 `static/style.css`，HTML `<head>` 引入 Google Fonts 即可。**不要修改 `:root` 变量名**，组件样式都依赖它们。

### B. Tailwind CSS v4

详见 `references/tailwind.md`。要点：在 `main.css` 用 `@theme` 块把 token 注册为 Tailwind 工具类（如 `bg-paper`、`text-ink-1`、`font-serif`），用 `@apply` 把组件基础类映射到这套配色。

### C. Nuxt / Vue / React

把 token 注入到项目的 CSS 入口（Nuxt 4 的 `app/assets/css/main.css`），组件用项目自带的 UI 库时，**优先级风格 = 清淡高雅**，自行覆写组件变量。

## 验证过的样板项目

`/Users/huhao/src/codesnip/python/ai/007-tts/static/` —— Kokoro TTS Web Demo，完整套用这套配色的实际项目。可作为对照参考。

## 关键陷阱（容易踩的坑）

1. **❌ 用纯黑 `#000` 做按钮或主文字** → 改为 `var(--ink-1)` 或 `var(--celadon-2)`
2. **❌ 用饱和蓝/紫/红做强调** → 改为青瓷绿或赭石
3. **❌ 大面积使用青瓷绿背景** → 它是点彩色，只用于按钮/选中/小圆点
4. **❌ 标题用无衬线** → 必须用衬线（思源宋体）才有书卷气
5. **❌ 强阴影（黑色、高模糊）** → 用极淡的暖色阴影（`rgba(76, 60, 30, 0.05)`）
6. **❌ 鲜艳的 emoji 当装饰** → 标题区不要用 🎙️ 这类 emoji，印章 logo 就够
7. **❌ 暗色模式** → 这套是亮色系。要做暗色需要单独设计，不能简单 invert

## 工作流：如何应用

1. **新建项目时** → 立刻复制 token 变量到 CSS 入口
2. **设计页面时** → 优先用组件样式速查表里的现成模式
3. **遇到不确定的颜色** → 优先选 `--ink-2/3/4`、`--card-2`、`--celadon-bg` 这类低饱和辅助色
4. **完成后** → 用 Chrome DevTools 截图看一眼实际效果，确认"清淡高雅"的感觉

## 参考文件

| 文件 | 用途 |
|---|---|
| `references/tokens.md` | 完整 token 详细说明 + 配色逻辑 |
| `references/base.css` | 一键复制的完整 CSS 基础（含 reset、组件、动画） |
| `references/tailwind.md` | Tailwind CSS v4 集成配置 + 实用工具类 |
| `references/components.md` | 按钮/表单/卡片/徽章/抽屉/历史记录 等组件完整样式 |
