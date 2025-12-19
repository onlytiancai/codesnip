# pnpm + Vite + Vue 3 + Tailwind CSS v4 + shadcn-vue + 黑白主题切换

> 适用环境：
>
> * Node ≥ 18
> * pnpm ≥ 8

---

## 一、创建 Vite + Vue 3 项目

```bash
pnpm create vite shadcn-vue-test
cd shadcn-vue-test
pnpm install
pnpm dev
```

选择：

* Framework：**Vue**
* Variant：**TypeScript**

确认能正常启动后继续。

---

## 二、安装 Tailwind CSS v4（⚠️ 关键）

### 1️⃣ 安装必须的依赖（v4 正确组合）

```bash
pnpm add -D tailwindcss @tailwindcss/postcss autoprefixer
```

> ❌ 不需要 `@tailwindcss/cli`
> ❌ 不需要 `tailwindcss init`

---

### 2️⃣ 手动创建配置文件（v4 没有 init）

#### `tailwind.config.js`

```js
import animate from 'tw-animate-css'

export default {
  darkMode: ['class'],
  content: [
    './index.html',
    './src/**/*.{vue,js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [animate],
}
```

---

#### `postcss.config.js`

```js
export default {
  plugins: {
    '@tailwindcss/postcss': {},
    autoprefixer: {},
  },
}
```

---

### 3️⃣ 使用 Vite 默认的 `style.css`

#### `src/style.css`

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

---

#### `src/main.ts`

```ts
import { createApp } from 'vue'
import App from './App.vue'
import './style.css'

createApp(App).mount('#app')
```

---

### 4️⃣ 安装动画插件（shadcn-vue 必需）

```bash
pnpm add -D tw-animate-css
```

---

## 三、配置路径别名（shadcn-vue 强制要求）

### 1️⃣ `vite.config.ts`

```ts
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
})
```

---

### 2️⃣ `tsconfig.json`（和 `tsconfig.app.json`）

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  }
}
```

> 如果有 `tsconfig.app.json`，**也要加同样内容**

重启 VS Code TS Server 或 `pnpm dev`。

---

## 四、初始化 shadcn-vue

```bash
pnpm add -D shadcn-vue
pnpm shadcn-vue init
```

关键选项：

```text
Framework: Vue
Global CSS file: src/style.css
Use CSS variables: Yes
Tailwind config: tailwind.config.js
Import alias: @/components
Utils alias: @/lib/utils
```

---

## 五、安装并验证组件

```bash
pnpm shadcn-vue add button
```

#### `App.vue`

```vue
<script setup lang="ts">
import { Button } from '@/components/ui/button'
</script>

<template>
  <Button>Hello shadcn-vue</Button>
</template>
```

✅ Button 正常渲染 → 一切 OK

---

## 六、为什么你看到 `text-green-500` “不明显”？

这是 **正常行为**：

* Tailwind v4 ✔ 正常
* shadcn-vue 使用 **语义化颜色系统**
* 推荐用：`text-primary` / `text-foreground`

例如：

```vue
<div class="text-4xl font-bold text-primary">
  Hello
</div>
```

---

## 七、🌓 黑白主题切换（核心目标）

### 1️⃣ 新建主题工具文件

#### `src/lib/theme.ts`

```ts
export function toggleDark() {
  document.documentElement.classList.toggle('dark')
}
```

---

### 2️⃣ 在组件中使用

#### `App.vue`

```vue
<script setup lang="ts">
import { Button } from '@/components/ui/button'
import { toggleDark } from '@/lib/theme'
</script>

<template>
  <div class="p-6 space-y-4">
    <Button @click="toggleDark">
      切换黑 / 白主题
    </Button>

    <div class="text-foreground text-xl">
      当前主题来自 shadcn-vue
    </div>
  </div>
</template>
```

🎉 点击按钮即可全站切换暗黑 / 明亮主题

---

## 八、（可选）记住用户主题偏好

```ts
export function initTheme() {
  if (localStorage.getItem('theme') === 'dark') {
    document.documentElement.classList.add('dark')
  }
}

export function toggleDark() {
  const root = document.documentElement
  const isDark = root.classList.toggle('dark')
  localStorage.setItem('theme', isDark ? 'dark' : 'light')
}
```

在 `main.ts`：

```ts
import { initTheme } from '@/lib/theme'
initTheme()
```

---

## 九、最终项目结构（对照）

```text
src/
 ├─ components/
 │   └─ ui/
 ├─ lib/
 │   ├─ utils.ts
 │   └─ theme.ts
 ├─ style.css
 ├─ App.vue
 └─ main.ts
tailwind.config.js
postcss.config.js
vite.config.ts
```

---

## 十、你现在已经完成了什么 🎉

* ✅ Tailwind CSS v4（正确方式）
* ✅ PostCSS v4 插件拆分
* ✅ shadcn-vue 官方推荐配置
* ✅ 语义化主题系统
* ✅ 黑白主题一键切换

