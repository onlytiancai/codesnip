# Tailwind CSS v4 集成

> 适用于：Tailwind CSS v4.x（CSS-first 配置）。v3 用法见底部补充。

## 1. 注册 token 到 `@theme`

在 `app/assets/css/main.css`（Nuxt 4）或全局 CSS 入口：

```css
@import "tailwindcss";

@theme {
  /* —— 颜色 —— */
  --color-paper:        #f5f1ea;
  --color-paper-2:      #faf7f1;
  --color-card:         #fdfbf6;
  --color-card-2:       #f8f4ec;
  --color-line:         #e8e1d3;
  --color-line-2:       #d9d0bd;

  --color-ink-1:        #2e2a24;
  --color-ink-2:        #5b5247;
  --color-ink-3:        #8a8273;
  --color-ink-4:        #b3aa9b;

  --color-celadon:      #6f7e6a;
  --color-celadon-2:    #5b6a56;
  --color-celadon-bg:   #e9eee5;
  --color-ochre:        #a07a5b;
  --color-ochre-bg:     #f1e9dd;
  --color-cinnabar:     #a8412e;
  --color-clay:         #b56b5a;
  --color-clay-bg:      #f4e3dd;
  --color-moss:         #5e7d65;
  --color-moss-bg:      #e3ebe2;

  /* —— 圆角 —— */
  --radius:             12px;
  --radius-sm:          8px;
  --radius-lg:          16px;

  /* —— 字体 —— */
  --font-sans:    -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                  "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  --font-serif:   "Noto Serif SC", "Songti SC", "STSong", serif;
  --font-mono:    "SF Mono", Menlo, Consolas, monospace;
}

@layer base {
  body {
    background-color: var(--color-paper);
    background-image:
      radial-gradient(1200px 600px at 20% 0%, rgba(241, 233, 219, 0.55), transparent 60%),
      radial-gradient(900px 500px at 90% 20%, rgba(232, 226, 211, 0.45), transparent 60%);
    background-attachment: fixed;
    color: var(--color-ink-1);
    line-height: 1.65;
  }
  ::selection { background: var(--color-celadon-bg); color: var(--color-celadon-2); }
}
```

注册后即可使用：
- `bg-paper` / `bg-card` / `bg-card-2` / `bg-celadon` / `bg-cinnabar` / ...
- `text-ink-1` / `text-celadon` / `text-ochre` / ...
- `border-line` / `border-celadon` / ...
- `font-sans` / `font-serif` / `font-mono`
- `rounded` / `rounded-sm` / `rounded-lg`

## 2. 加载思源宋体

`app/app.vue` 或全局 layout：

```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;500;600;700&display=swap" rel="stylesheet" />
```

## 3. 组件类（用 @apply 抽出来）

```css
@layer components {
  .panel {
    @apply bg-card border border-line rounded-lg p-6;
    box-shadow: 0 1px 2px rgba(76, 60, 30, 0.04), 0 8px 28px rgba(76, 60, 30, 0.05);
  }

  .panel-title {
    @apply font-serif text-base font-semibold text-ink-2 tracking-wider;
    @apply flex items-center gap-2 mt-6 mb-2;
  }
  .panel-title::before {
    content: "";
    @apply inline-block w-1.5 h-1.5 rounded-full bg-celadon opacity-75;
  }
  .panel-title:first-child { @apply mt-0; }
  .panel-title:first-child::before { display: none; }

  .btn {
    @apply px-5 py-2.5 rounded-sm text-sm font-medium tracking-wide;
    @apply transition-all duration-200 border;
  }
  .btn-primary {
    @apply bg-celadon text-[#f7f4ee] border-celadon font-semibold;
  }
  .btn-primary:hover:not(:disabled) {
    @apply bg-celadon-2 border-celadon-2 -translate-y-px;
    box-shadow: 0 6px 18px rgba(95, 110, 88, 0.22);
  }
  .btn-secondary {
    @apply bg-card text-ink-1 border-line-2;
  }
  .btn-secondary:hover:not(:disabled) {
    @apply bg-celadon-bg border-celadon text-celadon-2;
  }
  .btn-ghost {
    @apply bg-transparent text-ink-2 border-line;
  }
  .btn-ghost:hover:not(:disabled) {
    @apply bg-card-2 text-ink-1;
  }

  .field {
    @apply w-full px-4 py-3.5 bg-paper-2 border border-line rounded-sm;
    @apply text-ink-1 text-base leading-relaxed transition-all duration-200;
  }
  .field:focus {
    @apply outline-none bg-card border-celadon;
    box-shadow: 0 0 0 3px var(--color-celadon-bg);
  }

  .badge {
    @apply px-2 py-0.5 rounded bg-card-2 border border-line text-ink-3 text-xs;
  }
  .badge-lang {
    @apply bg-celadon-bg text-celadon-2 border-transparent;
  }

  /* 印章 logo */
  .seal {
    @apply inline-flex items-center justify-center w-11 h-11 bg-cinnabar text-[#f7ecd8];
    @apply font-serif font-bold text-base rounded;
    box-shadow:
      0 0 0 2px #f7ecd8,
      0 0 0 3px var(--color-cinnabar),
      0 4px 10px rgba(168, 65, 46, 0.18);
    transform: rotate(-3deg);
    line-height: 1;
  }
}
```

## 4. 实用类用法示例

```html
<div class="container">
  <section class="panel">
    <h2 class="panel-title">输入文本</h2>
    <textarea class="field" rows="8"></textarea>

    <h2 class="panel-title">操作</h2>
    <button class="btn btn-primary">提交</button>
    <button class="btn btn-secondary">取消</button>
    <button class="btn btn-ghost">更多</button>
  </section>
</div>
```

## 5. 标题区示例

```html
<header class="text-center pt-14 pb-10 px-6 border-b border-line relative">
  <div class="absolute top-6 left-1/2 -translate-x-1/2 w-15 h-px bg-line-2"></div>
  <h1 class="flex items-center justify-center gap-3.5 font-serif text-4xl font-semibold tracking-wide text-ink-1">
    <span class="seal">聲</span>
    <span>App Name</span>
  </h1>
  <p class="text-ink-3 text-base">副标题描述</p>
</header>
```

## 6. Tailwind v3 补充

v3 不支持 `@theme`，需要在 `tailwind.config.js` 里注册：

```js
module.exports = {
  theme: {
    extend: {
      colors: {
        paper:       '#f5f1ea',
        'paper-2':   '#faf7f1',
        card:        '#fdfbf6',
        'card-2':    '#f8f4ec',
        line:        '#e8e1d3',
        'line-2':    '#d9d0bd',
        'ink-1':     '#2e2a24',
        'ink-2':     '#5b5247',
        'ink-3':     '#8a8273',
        'ink-4':     '#b3aa9b',
        celadon:     '#6f7e6a',
        'celadon-2': '#5b6a56',
        'celadon-bg':'#e9eee5',
        ochre:       '#a07a5b',
        cinnabar:    '#a8412e',
        clay:        '#b56b5a',
        moss:        '#5e7d65',
      },
      fontFamily: {
        sans:  ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'PingFang SC', 'sans-serif'],
        serif: ['"Noto Serif SC"', '"Songti SC"', '"STSong"', 'serif'],
        mono:  ['"SF Mono"', 'Menlo', 'Consolas', 'monospace'],
      },
      borderRadius: { 'sm': '8px', DEFAULT: '12px', 'lg': '16px' },
    },
  },
}
```

## 7. 常见 Nuxt UI v4 覆写

如果项目用了 Nuxt UI（v3/v4），需要把它的默认色板（`primary: green`，`neutral: slate`）改成本配色：

```ts
// app/app.config.ts
export default defineAppConfig({
  ui: {
    primary: 'celadon',     // 自定义颜色，会在 main.css @theme 里注册
    gray:    'ink',
    icons: { dynamic: true },
  },
})
```

由于 Nuxt UI 用 oklch 调色板，自定义颜色需要额外写一段 oklch 值映射。最简单的方案是：

```css
/* 在 main.css 里 */
:root {
  --ui-primary: var(--color-celadon);
  --ui-text:    var(--color-ink-1);
  --ui-bg:      var(--color-card);
  --ui-bg-elevated: var(--color-card-2);
  --ui-border:  var(--color-line);
}
```

然后在 `<UButton color="primary">` 等组件上会自然使用青瓷绿。
