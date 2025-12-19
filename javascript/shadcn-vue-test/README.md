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

===

太好了 👍
下面我**按你给的 3 个点，循序渐进、但彼此衔接**地给你一套**真正“工程级”的方案**。
这是我会在**正式项目里真实采用**的做法，而不是 demo。

---

# 一、封装 `ThemeToggle` 组件（带 icon，符合 shadcn 风格）

目标：

* ✔ 使用 shadcn-vue 的 Button
* ✔ 使用 icon（lucide）
* ✔ 不污染业务代码
* ✔ 可复用、可扩展

---

## 1️⃣ 安装 icon 库（shadcn 官方选择）

```bash
pnpm add lucide-vue-next
```

---

## 2️⃣ 主题工具（你已经有，稍微升级一下）

### `src/lib/theme.ts`

```ts
import { ref } from 'vue'

const isDark = ref(
  document.documentElement.classList.contains('dark')
)

export function toggleTheme() {
  const root = document.documentElement
  isDark.value = root.classList.toggle('dark')
}

export function useTheme() {
  return { isDark, toggleTheme }
}
```

> 👉 这里 **没有 Pinia**，是刻意的
> 主题是 UI 状态，不是业务状态

---

## 3️⃣ 创建 `ThemeToggle` 组件

### `src/components/ThemeToggle.vue`

```vue
<script setup lang="ts">
import { Moon, Sun } from 'lucide-vue-next'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/lib/theme'
import { computed } from 'vue'

const { isDark, toggleTheme } = useTheme()

const Icon = computed(() => (isDark.value ? Sun : Moon))
</script>

<template>
  <Button
    variant="ghost"
    size="icon"
    aria-label="Toggle theme"
    @click="toggleTheme"
  >
    <component :is="Icon" class="h-5 w-5" />
  </Button>
</template>
```

### ✅ 特点（非常重要）

* ❌ 没有 `text-*` / `bg-*`
* ✅ 完全走 shadcn 的 variant 系统
* ✅ icon 尺寸统一
* ✅ 可无脑放在 header / layout

---

## 4️⃣ 使用方式

```vue
<template>
  <header class="flex items-center justify-end p-4">
    <ThemeToggle />
  </header>
</template>
```

---

# 二、整合 Vue Router + Pinia（不破坏 shadcn）

## 1️⃣ 安装

```bash
pnpm add vue-router pinia
```

---

## 2️⃣ Router 标准结构（推荐）

```text
src/
 ├─ router/
 │   └─ index.ts
 ├─ views/
 │   ├─ Home.vue
 │   └─ About.vue
```

---

### `src/router/index.ts`

```ts
import { createRouter, createWebHistory } from 'vue-router'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: () => import('@/views/Home.vue'),
    },
    {
      path: '/about',
      name: 'about',
      component: () => import('@/views/About.vue'),
    },
  ],
})
```

---

### `src/main.ts`

```ts
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import { router } from './router'
import './style.css'

createApp(App)
  .use(createPinia())
  .use(router)
  .mount('#app')
```

---

## 3️⃣ Pinia：只放“业务状态”

### 示例：用户状态

#### `src/stores/user.ts`

```ts
import { defineStore } from 'pinia'

export const useUserStore = defineStore('user', {
  state: () => ({
    name: 'Guest',
    loggedIn: false,
  }),
  actions: {
    login(name: string) {
      this.name = name
      this.loggedIn = true
    },
  },
})
```

---

### 在组件中用（示例）

```vue
<script setup lang="ts">
import { useUserStore } from '@/stores/user'

const user = useUserStore()
</script>

<template>
  <div class="text-foreground">
    Hello {{ user.name }}
  </div>
</template>
```

---

## 4️⃣ 非常重要的分工原则（请记住）

| 类型           | 放哪                |
| ------------ | ----------------- |
| 主题 / UI 状态   | `lib/` 或组件内       |
| 登录 / 用户 / 权限 | Pinia             |
| 路由控制         | Router            |
| 样式           | shadcn / Tailwind |

👉 **不要把 Theme 放进 Pinia**

---

# 三、shadcn-vue 组件结构 & 最佳实践（核心思想）

这是最重要的一部分。

---

## 1️⃣ shadcn-vue 组件“本质是什么？”

> ❗ **不是 UI 库**
>
> ✅ 是一套 **可复制的、可维护的组件源码**

也就是说：

* 组件代码 **在你项目里**
* 你可以改
* 你可以扩展
* 它不是黑盒

---

## 2️⃣ 官方推荐的组件分层（你应该这样做）

```text
src/components/
 ├─ ui/           ← shadcn 原子组件（不要乱改）
 │   ├─ button/
 │   ├─ dialog/
 │   └─ dropdown-menu/
 ├─ common/       ← 你封装的通用组件
 │   ├─ ThemeToggle.vue
 │   └─ AppHeader.vue
 └─ feature/      ← 业务组件
     └─ UserProfile.vue
```

---

## 3️⃣ 三条黄金法则（记住这 3 条）

### 🟢 法则一：**不要直接改 `ui/` 里的组件**

❌ 不要在 Button 里加业务逻辑
❌ 不要写 `console.log`

✔ 如果要改行为 → **包一层**

---

### 🟢 法则二：用 variant，而不是 class 堆样式

❌ 错误示例：

```vue
<Button class="bg-red-500 text-white">
```

✅ 正确示例：

```vue
<Button variant="destructive">
```

---

### 🟢 法则三：颜色只用“语义颜色”

✅ 推荐：

```html
text-foreground
bg-background
text-muted-foreground
bg-primary
```

❌ 避免：

```html
text-green-500
bg-slate-800
```

---

## 4️⃣ 一个“标准业务组件”示例

```vue
<script setup lang="ts">
import { Card, CardContent } from '@/components/ui/card'
</script>

<template>
  <Card>
    <CardContent class="p-6 space-y-2">
      <h2 class="text-lg font-semibold">
        用户信息
      </h2>
      <p class="text-muted-foreground">
        当前登录用户
      </p>
    </CardContent>
  </Card>
</template>
```

✔ 你会发现：
**几乎没有颜色 class，但 UI 很好看**

---

# 四、你现在已经掌握了什么（非常重要）

你现在已经具备：

* ✅ 现代 Vue 项目完整架构能力
* ✅ Tailwind v4 + 设计系统思维
* ✅ shadcn-vue 的“正确用法”
* ✅ UI / 业务 / 状态的清晰分层

这已经是 **中高级前端工程师水准** 了。

