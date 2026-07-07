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

## 录制成视频（mp4）

`pnpm record` 启动 Playwright（headed Chromium）自动驱动 v-click，等每段旁白音频播完再点下一次，录完用 ffmpeg 合成 1280×720 / 30fps / h264+aac 的 mp4。**不写死 v-click 数量**，脚本通过「`__playCount` 是否变化 + 当前 slide URL 是否变化」动态判断翻页和结束。

### 用法

```bash
pnpm record                              # → output/slide-1.mp4（默认 720p）
pnpm record --width 1920 --height 1080   # 临时切回 1080p
pnpm record --out my-video.mp4           # 指定输出
pnpm record --keep-server                # 录完保留 dev server，方便调试
pnpm record --no-clean                   # 复用 build/ 里的中间产物
HEADLESS=1 pnpm record                   # 无 GUI 环境用 headless（音频会空）
```

`--width` / `--height` 必须是正整数；分辨率会传给 Playwright 的 viewport 和 recordVideo，ffmpeg 按源尺寸直接 mux，不会重新缩放。

### 中间产物（在 `build/` 下）

| 文件 | 来源 |
|---|---|
| `raw.webm` | Playwright 录的 webm 视频（vp8/vp9，无音轨） |
| `narration.webm` | 页面内 MediaRecorder 抓的 webm 音频（opus） |
| `narration-padded.m4a` | 在音频前垫静音对齐视频开头 + 转 aac m4a |
| `output/<name>.mp4` | 最终合成产物 |

### 工作流程

1. `pnpm dev --port 3030` 起 dev server（轮询 `localhost` → `127.0.0.1` → `[::1]` 命中即可，绕过 macOS 上 IPv6/IPv4 解析顺序导致的 fetch 失败）。
2. `chromium.launch({ headless: false })` 开真窗口 + `addInitScript` 注入一段脚本：patch `window.Audio`，把每个 `<audio>` 节点同时连到扬声器和 `MediaStreamAudioDestinationNode`，由 `MediaRecorder` 录成音频块。
3. 主循环：按 `Space` → 250ms 后 `evaluate` 读 `__playCount`；有变化就 `waitForFunction(() => __lastAudio.ended === true)` 等播完（timeout 60s），没变化说明当前 slide 的 v-click 已点完，按 `ArrowRight` 切页，URL 不再变就结束。
4. 收尾：浏览器侧停 MediaRecorder → 回传字节数组到 Node 写 `narration.webm`；关 page 拿 `raw.webm`；用 `ffmpeg` 给音频前垫 `__firstPlayTimeMs` 时长静音（首次 `audio.play` 距 MediaRecorder 启动的偏移），再 mux 成 mp4。
5. 验证：ffprobe 检查 h264（默认 1280×720，可用 `--width`/`--height` 覆盖）+ aac 双轨、时长对齐。

### 关键坑

- **必须 headed**：macOS / Linux 上 headless Chromium 经常录不到音频；脚本默认 `headless: false`，会弹一个真浏览器窗口（首次按 `Space` 顺便当 user gesture 放行 autoplay）。`HEADLESS=1` 兜底无 GUI 环境，但此时 mp4 会无声。
- **macOS 上 `127.0.0.1` 可能 fetch 失败**：`pnpm dev` 默认绑 `localhost`，脚本优先用 `localhost`，不命中再退到 `127.0.0.1` 和 `[::1]`。不要给 dev 加 `--host 127.0.0.1`（Slidev 52.x 不支持该 flag）。
- **音视频同步靠性能时间戳**：MediaRecorder 在首次 `audio.play()` 时才有数据，视频从 context 创建就开始录。脚本用 `performance.now()` 算偏移再垫静音，不靠 `Date.now()`（避免多 tab / 挂起漂移）。
- **合成前音频要转 aac m4a**：webm 容器不支持 aac 编码，ffmpeg 会在 mux 阶段报 "Only VP8/VP9/AV1 video and Vorbis/Opus audio and WebVTT subtitles are supported for WebM"；`padAudio` 已统一输出 `.m4a`。
- **每次入口默认清空 `build/`**：避免 `raw.webm` 累积导致 ffmpeg 多输入冲突；想调试时加 `--no-clean`。
- **录制时不要编辑 `slides.md` / `components/`**：dev server 的 HMR 会触发 reload，中断录音。

### 人工裁剪视频前面的白屏部分

`pnpm record` 输出后，终端里会有一行类似：

```
[main] [参考] h1+fonts.ready 时刻 = 1697ms（从 __recorderStart 算起，可用于 ffmpeg -ss 手动裁剪）
```

把这个值（秒）填到 `-ss`：

    ffmpeg -ss 1.7 -i output/slide-1.mp4 -c copy output/slide-1-trimmed.mp4

- `-ss 1.7`：从第 1.7 秒开始（输入侧 seek）
- `-c copy`：流复制，不重新编码，几秒完成

**微调**：视频录制起点（`newContext`）比 `__recorderStart` 早 100~500ms，所以日志值通常会留 ~200ms 加载空白。打开裁好的视频看一眼：

- 还有白屏 → `-ss` 加大 0.1~0.2
- 开头画面已经动起来 → `-ss` 减小 0.1~0.2
- 第一句旁白被切了 → 减小 0.3~0.5（旁白位置 = `firstPlayTimeMs/1000 - ss`）

注意：

- `-ss` 放在 `-i` 前是快速但按关键帧对齐的 seek。H.264 / AAC 流对 fast seek 都安全，所以 `-c copy` 不需要重编码。
- 如果想要逐帧精度，把 `-ss` 放到 `-i` 后（`-i input -ss 1.7 -c copy`），但会触发解码，耗时。

如果想要重新编码保证精度：

    ffmpeg -i output/slide-1.mp4 -ss 1.7 -c:v libx264 -c:a aac output/slide-1-trimmed.mp4

想要精确且保留时间戳（慢但准）：

    ffmpeg -i output/slide-1.mp4 -ss 1.7 -c copy -avoid_negative_ts make_zero output/slide-1-trimmed.mp4