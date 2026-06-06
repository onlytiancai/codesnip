# 组件样式库

「茶韵 · 宋雅」配色系统的完整组件实现。每个组件都基于 `base.css` 的 token，所有 `--var()` 引用都可直接复用。

---

## 1. 头部 Hero

```html
<header class="hero">
  <div class="hero-inner">
    <div class="hero-title">
      <span class="seal" aria-hidden="true">聲</span>
      <span>App Name</span>
    </div>
    <p class="subtitle">副标题描述，可以包含 <code>inline code</code></p>
  </div>
</header>
```

样式：见 `base.css` 的 `.hero`、`.hero-title`、`.seal`、`.subtitle` 段。

**变体**：
- 居中 → `text-align: center`（默认）
- 左对齐 → 移除 `text-align: center`，加 `text-align: left; padding-left: 4rem;`
- 双标题 → `.hero-title` 后面加一个 `<p class="hero-subtitle-en">English Subtitle</p>`

---

## 2. 面板 Panel

```html
<section class="panel">
  <h2>面板标题</h2>
  <p>面板内容...</p>
</section>
```

**多个 panel 并排**：
```html
<main class="container">
  <section class="panel">...</section>
  <section class="panel">...</section>
</main>
```

`.container` 已是 grid 布局，移动端自动单列。

**Panel 变体**：
- 默认 → 象牙白 + 淡边 + 极淡阴影
- 凹陷 → 加 `panel-inset` 类，背景 `var(--paper-2)`，无阴影

---

## 3. 按钮 Button

```html
<button class="primary">主操作</button>
<button class="secondary">次操作</button>
<button class="ghost">轻操作</button>
<button class="primary" disabled>禁用</button>
```

按钮组（推荐间距）：
```html
<div class="actions">
  <button class="primary">保存</button>
  <button class="secondary">取消</button>
  <button class="ghost">删除</button>
</div>
```

```css
.actions { display: flex; gap: 0.6rem; margin-top: 1.5rem; flex-wrap: wrap; }
```

**图标按钮**（如下载）：
```html
<button class="secondary">
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>
  </svg>
  下载
</button>
```
按钮内 SVG 用 `currentColor` 即可继承按钮文字色。

---

## 4. 表单 Form

### 文本输入
```html
<input type="text" placeholder="请输入..." />
<textarea rows="6" placeholder="多行文本..."></textarea>
```

### 搜索框
```html
<div class="voice-filter">
  <input type="search" placeholder="搜索..." />
  <select><option>所有</option></select>
</div>
```
```css
.voice-filter { display: flex; gap: 0.55rem; }
.voice-filter input { flex: 1; }
```

### 单选 / 复选
```html
<label><input type="radio" name="fmt" value="wav" checked /> WAV</label>
<label><input type="radio" name="fmt" value="mp3" /> MP3</label>
```
label 容器：
```css
.format-row { display: flex; gap: 1.6rem; }
.format-row label { display: inline-flex; align-items: center; gap: 0.45rem; cursor: pointer; }
```

### 滑块
```html
<div class="speed-row">
  <input type="range" min="0.5" max="2.0" step="0.05" value="1.0" />
  <span class="speed-value">1.00×</span>
</div>
```
```css
.speed-row { display: flex; align-items: center; gap: 1rem; }
.speed-value { font-family: var(--font-mono); min-width: 64px; text-align: right; color: var(--celadon-2); }
```

### 状态指示器
```html
<div class="meta">
  <span id="count">42</span> 字符
  <span class="dot">·</span>
  <span class="status-idle">就绪</span>
  <!-- 状态切换：status-loading / status-ok / status-error -->
</div>
```
```css
.status-idle    { color: var(--ink-3); }
.status-loading { color: var(--ochre); }
.status-ok      { color: var(--moss); }
.status-error   { color: var(--clay); }

.status-loading::before {
  content: "";
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: currentColor;
  margin-right: 0.4em;
  vertical-align: 0.12em;
  animation: inkPulse 1.6s ease-in-out infinite;
}
```

---

## 5. 卡片 / 网格（音色选择类）

```html
<div class="card-grid">
  <div class="card">普通卡片</div>
  <div class="card selected">选中卡片</div>
  <div class="card">普通卡片</div>
</div>
```

```css
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
  gap: 0.55rem;
  max-height: 340px;
  overflow-y: auto;
  padding: 0.3rem;
}

.card {
  padding: 0.65rem 0.8rem;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all 0.2s ease;
  user-select: none;
}

.card:hover {
  background: var(--card-2);
  border-color: var(--celadon);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

.card.selected {
  background: var(--celadon);
  border-color: var(--celadon);
  box-shadow: 0 4px 14px rgba(95, 110, 88, 0.25);
}
.card.selected .card-name,
.card.selected .card-label { color: #f7f4ee; }
.card.selected .card-label { color: rgba(247, 244, 238, 0.78); }
.card.selected .badge { background: rgba(247, 244, 238, 0.18); color: #f7f4ee; }
.card.selected .badge.lang { background: rgba(247, 244, 238, 0.28); }
```

---

## 6. 徽章 Badge

```html
<span class="badge">标签</span>
<span class="badge lang">语言</span>
```

```css
.badge {
  background: var(--card-2);
  padding: 0.1rem 0.45rem;
  border-radius: 4px;
  color: var(--ink-3);
  border: 1px solid var(--line);
  font-size: 0.7rem;
}
.badge.lang {
  background: var(--celadon-bg);
  color: var(--celadon-2);
  border-color: transparent;
}
```

**变体**：
- 重要：背景 `var(--cinnabar)` + 白字
- 警告：背景 `var(--clay-bg)` + `var(--clay)` 字
- 成功：背景 `var(--moss-bg)` + `var(--moss)` 字

---

## 7. 播放 / 媒体

```html
<div class="player">
  <audio controls></audio>
  <div class="player-info">尚未合成</div>
</div>
```

```css
.player {
  background: var(--paper-2);
  padding: 1rem 1.1rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--line);
  transition: border-color 0.2s;
}
.player:hover { border-color: var(--line-2); }
.player audio { width: 100%; border-radius: 6px; }
.player-info {
  font-size: 0.83rem;
  color: var(--ink-3);
  margin-top: 0.6rem;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.player-info::before {
  content: "♪";
  color: var(--celadon);
  font-size: 0.95rem;
}
```

---

## 8. 抽屉 / 侧边栏

```html
<aside class="drawer">
  <div class="drawer-header">
    <h3>标题</h3>
    <button class="ghost">关闭</button>
  </div>
  <ul class="drawer-list">
    <li>...</li>
  </ul>
</aside>
```

```css
.drawer {
  position: fixed;
  top: 0; right: 0;
  height: 100vh;
  width: 380px;
  background: var(--card);
  border-left: 1px solid var(--line);
  box-shadow: var(--shadow-lg);
  z-index: 100;
  display: flex;
  flex-direction: column;
  animation: drawerIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.drawer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.1rem 1.3rem;
  border-bottom: 1px solid var(--line);
  background: var(--card-2);
}
.drawer-header h3 {
  margin: 0;
  font-family: var(--font-serif);
  font-weight: 600;
  font-size: 1.05rem;
  letter-spacing: 0.04em;
}

.drawer-list {
  list-style: none;
  margin: 0;
  padding: 0.7rem;
  overflow-y: auto;
  flex: 1;
}

@keyframes drawerIn {
  from { transform: translateX(20px); opacity: 0; }
  to   { transform: translateX(0);    opacity: 1; }
}
```

---

## 9. 表格（数据展示）

```html
<table class="data-table">
  <thead>
    <tr><th>名称</th><th>类型</th><th>时间</th></tr>
  </thead>
  <tbody>
    <tr><td>item-1</td><td><span class="badge lang">中文</span></td><td>2026-06-06</td></tr>
  </tbody>
</table>
```

```css
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
}
.data-table th {
  text-align: left;
  font-family: var(--font-serif);
  font-weight: 600;
  color: var(--ink-2);
  padding: 0.6rem 0.8rem;
  border-bottom: 1px solid var(--line-2);
  letter-spacing: 0.04em;
}
.data-table td {
  padding: 0.7rem 0.8rem;
  border-bottom: 1px solid var(--line);
  color: var(--ink-1);
}
.data-table tr:hover td { background: var(--card-2); }
```

---

## 10. 模态框（弹窗）

```html
<div class="modal-mask">
  <div class="modal">
    <h3 class="modal-title">标题</h3>
    <div class="modal-body">内容...</div>
    <div class="modal-footer">
      <button class="ghost">取消</button>
      <button class="primary">确定</button>
    </div>
  </div>
</div>
```

```css
.modal-mask {
  position: fixed; inset: 0;
  background: rgba(46, 42, 36, 0.4);
  display: flex; align-items: center; justify-content: center;
  z-index: 200;
  animation: fadeIn 0.2s ease;
}
.modal {
  background: var(--card);
  border-radius: var(--radius-lg);
  padding: 1.6rem 1.8rem;
  box-shadow: var(--shadow-lg);
  max-width: 480px;
  width: 90%;
  animation: modalIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.modal-title {
  margin: 0 0 1rem;
  font-family: var(--font-serif);
  font-size: 1.15rem;
  color: var(--ink-1);
}
.modal-body { color: var(--ink-2); font-size: 0.95rem; }
.modal-footer {
  display: flex; justify-content: flex-end; gap: 0.6rem;
  margin-top: 1.5rem;
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes modalIn {
  from { transform: translateY(-12px); opacity: 0; }
  to   { transform: translateY(0);     opacity: 1; }
}
```

---

## 11. 进度条

```html
<div class="progress"><div class="progress-bar" style="width: 60%"></div></div>
```

```css
.progress {
  width: 100%; height: 4px;
  background: var(--card-2);
  border-radius: 2px;
  overflow: hidden;
}
.progress-bar {
  height: 100%;
  background: var(--celadon);
  border-radius: 2px;
  transition: width 0.3s ease;
}
```

---

## 12. 分页 / 面包屑

```html
<nav class="breadcrumb">
  <a href="/">首页</a>
  <span class="sep">›</span>
  <a href="/users">用户</a>
  <span class="sep">›</span>
  <span class="current">张三</span>
</nav>
```

```css
.breadcrumb { font-size: 0.9rem; color: var(--ink-3); }
.breadcrumb a { color: var(--ochre); }
.breadcrumb a:hover { color: var(--celadon-2); }
.breadcrumb .sep { margin: 0 0.5rem; color: var(--ink-4); }
.breadcrumb .current { color: var(--ink-1); font-weight: 500; }
```

---

## 速查：哪个组件用什么颜色

| 场景 | 颜色 token |
|---|---|
| 主操作按钮 | `bg-celadon text-#f7f4ee` |
| 主操作按钮 hover | `bg-celadon-2` + 暖色阴影 |
| 次要按钮 | `bg-card border-line-2` |
| 次要按钮 hover | `bg-celadon-bg border-celadon text-celadon-2` |
| 选中项 | `bg-celadon text-#f7f4ee` + 暖色阴影 |
| 链接 | `text-ochre` |
| 链接 hover | `text-celadon-2` |
| 输入框聚焦 | `border-celadon + 3px celadon-bg 晕` |
| 错误提示 | `bg-clay-bg text-clay border-#e8c9bf` |
| 成功提示 | `bg-moss-bg text-moss` |
| 加载中文字 | `text-ochre + inkPulse 动画` |
| 装饰性印章 | `bg-cinnabar` + 暖色阴影 |
| 面板标题前圆点 | `bg-celadon opacity-75` |
