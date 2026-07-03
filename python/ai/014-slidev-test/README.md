# Slidev v-click 语音旁白

一个 Slidev 幻灯片示例：**每次 v-click 触发时，自动播放对应的中文语音旁白**。
音频离线批量生成（跑脚本调 MiniMax TTS），在线播放由一个订阅导航状态的 Vue 组件完成。

## 效果

- 进入标题页 → 静默
- 按 `→` 触发 v-click 1 → 自动播 `slide-1-click-1.mp3`（“什么是泛型……”）
- 再按 `→` 触发 v-click 2 → 自动播 `slide-1-click-2.mp3`（“基础泛型函数”）
- 切走当前 slide → 音频立即停止

## 快速开始

```bash
pnpm install

# 生成语音（需要 MINIMAX_API_KEY）
export MINIMAX_API_KEY=xxxx
pnpm tts:slides -- --dry   # 先干跑，确认抽取到的旁白文本正确
pnpm tts:slides            # 真正合成，写入 public/audio/

# 启动
pnpm dev                   # http://localhost:3030
```

## 旁白语法

旁白文本在 `slides.md` 里**显式定义**（不再从 v-click 内容里猜），
在每个 `<div v-click>` 块内写一行 HTML 注释指令：

```html
<div v-click>

## 什么是泛型？

泛型允许你创建**参数化类型**……

<!-- narrate: 什么是泛型？泛型允许你创建参数化类型，在定义函数、接口、类时不指定具体类型，而在使用时再确定 -->

</div>
```

规则：

- 每个顶层 `<div v-click>` 记为一个 click，序号从 1 起（与 Slidev 实际 click index 对齐）
- 块内的 `<!-- narrate: ... -->` 即该 click 的旁白；**没有则该 click 无音频**
- 注释不渲染、不污染演讲者备注（后面还有 `</div>`，不是 slide 尾部注释）
- 代码围栏（``` / ~~~）内整段忽略，示例代码里的注释不会被误当旁白

## 架构

```
slides.md ──(离线抽取)──► scripts/extract-clicks.ts ──► scripts/tts-slides.ts ──► public/audio/*.mp3
                                                                                        │
                                                                                        ▼ (在线播放)
                                        global-bottom.vue ──► components/ClickAudio.vue ─┘
```

| 文件 | 作用 |
|---|---|
| `scripts/extract-clicks.ts` | 栈式 fence/div 状态机解析 `slides.md`，读出每个 `<div v-click>` 块内的 `<!-- narrate: ... -->` 指令作为旁白，丢弃代码块 |
| `scripts/tts-slides.ts` | CLI：调 `extract-clicks` + `synthesize()`，把 mp3 写到 `public/audio/slide-N-click-M.mp3` |
| `scripts/minimax-tts.ts` | MiniMax TTS API 封装（`synthesize()`） |
| `components/ClickAudio.vue` | 播放组件，watch 导航状态，命中映射就播，切走/无匹配则停 |
| `global-bottom.vue` | Slidev 全局层，硬编码 v-click → 音频映射，挂载 `ClickAudio` |

`slides.md` 的内容不受影响：旁白写在 HTML 注释里，不渲染、不侵入视觉；音频能力通过全局层挂载。

## 生成脚本用法

```bash
pnpm tts:slides                       # 生成所有缺失片段（已存在则跳过）
pnpm tts:slides -- --dry              # 只打印计划，不调用 API
pnpm tts:slides -- --force            # 覆盖已存在的 mp3
pnpm tts:slides -- --slide 1 --click 1  # 只生成指定单段
```

网络失败兜底（走代理）：

```bash
HTTPS_PROXY=http://127.0.0.1:10808 pnpm tts:slides
```

## 关键坑（务必注意）

- **全局层里 `useSlideContext()` 的 `$clicks` 是死的**：在 `global-bottom.vue` / `ClickAudio.vue` 这类全局层组件里，`useSlideContext()` 返回的 `$clicks` 只有挂载时的初值、不随按键更新。必须改用 `useNav()` 的 `currentPage` / `clicks`，它们才是随导航实时更新的全局响应式值。
- **用 `onSlideLeave` 而非 `onUnmounted` 清理**：Slidev 组件实例常驻，`onUnmounted` 不会在切 slide 时触发。此外 `ClickAudio` 里 `playFor` 遇到「无匹配」也会 `stop()`，作为切走停止的兜底。
- **浏览器 autoplay 拦截**：首次 `→` 即用户手势，之后放行；被拦截时组件会 `console.warn` 而非报错。
- **旁白抽取用栈式 fence/div 配平**：不能复用渲染端正则，fence 围栏内代码整段丢弃；旁白只认块内的 `<!-- narrate: ... -->` 指令，不再从内容里猜文本。
- **`@slidev/parser` 只在 Node 侧可用**：其 `parseSync` 依赖 fs，浏览器里跑不了，所以 `global-bottom.vue` 的映射表选择硬编码。

## 增改 v-click 后的同步步骤

1. 编辑 `slides.md`，给新的 `<div v-click>` 块加 `<!-- narrate: ... -->` 旁白
2. `pnpm tts:slides -- --dry` 确认抽取文本正确
3. `pnpm tts:slides` 生成新 mp3
4. 手动同步 `global-bottom.vue` 里的 `items` 映射表
