# 视频录制原理（`scripts/record-video.ts`）

> 把 Slidev 演示录成 mp4 的整条管线拆解。配套命令：`pnpm record`。

## 全景图

```
┌───────────────────────── 浏览器内（headed Chromium）─────────────────────────┐
│                                                                             │
│   <audio> 元素                                                              │
│      │                                                                      │
│      ├─► AudioContext.destination  ──► 扬声器（用户照常听到）                │
│      │                                                                      │
│      └─► MediaStreamDestination  ──► MediaRecorder  ──► narration.webm       │
│                                                                             │
│   Playwright recordVideo ─────────────────────────────────► raw.webm         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │                            │
                              ▼                            ▼
                  ┌──────────────────────┐    ┌──────────────────────────┐
                  │ ffmpeg 垫静音        │    │ ffmpeg mux               │
                  │ narration.webm       │    │   raw.webm               │
                  │   + anullsrc         │──► │   + narration-padded.m4a │
                  │   → narration-       │    │   → slide-1.mp4          │
                  │     padded.m4a       │    │   (h264 + aac)           │
                  └──────────────────────┘    └──────────────────────────┘
```

四个独立环节：**音频分流 → 画面录制 → 翻页自动化 → 音画 mux**。

---

## 1. 双轨录制：画面与声音分开走

视频和音频来自两个不同来源，最终 mux 到 mp4。

| 通道 | 来源 | 容器 / 编码 |
|---|---|---|
| 视频 | Playwright `recordVideo` 截浏览器视口 | webm (vp8/vp9) |
| 音频 | 页面内 `MediaRecorder` 抓 `MediaStream` | webm (opus) |

### 为什么必须 headed

macOS / Linux 上 headless Chromium 的音频设备常被沙箱隔离，录出来是空流。脚本默认 `headless: false` 弹一个真窗口 —— 顺带首次按 `Space` 还能当 user gesture 放行浏览器的 autoplay 拦截。`HEADLESS=1` 是给 CI / 无 GUI 环境的兜底，但此时 mp4 会无声。

---

## 2. 声音怎么"偷"出来：Audio 钩子

`browserContext.addInitScript` 在每个新文档加载前注入这段（节选）：

```js
const ctx = new AudioContext();
const dest = ctx.createMediaStreamDestination();

// 替换 window.Audio 构造器
const PatchedAudio = function (src) {
  const a = new OrigAudio(src);
  const srcNode = ctx.createMediaElementSource(a);
  srcNode.connect(ctx.destination);   // 扬声器
  srcNode.connect(dest);              // 录制支路
  return a;
};
window.Audio = PatchedAudio;

new MediaRecorder(dest.stream, { mimeType: 'audio/webm;codecs=opus' }).start(250);
```

原理：每个 `<audio>` 元素既连到扬声器让用户听到声音，又同时灌进 `MediaStreamDestinationNode`。`MediaRecorder` 把这个流持续压成 webm/opus 块，每 250ms 触发一次 `dataavailable` 事件，Node 端在结束时把 `__audioChunks` 拼接回写文件。

**关键点**：
- 注入点在 `addInitScript`，每个新文档都自动生效
- 用 `createMediaElementSource` 而不是 `createMediaStreamSource` —— 前者能拿到元素粒度的 `ended` 事件
- 不影响实际播放体验，**用户照常从扬声器听到声音**

---

## 3. 翻页与结束判定：两个状态机

脚本不写死 v-click 数量，靠两个观察量动态判断：

```
                    ┌─────────────────────────────┐
                    │  while (true)               │
                    │    按 Space                 │
                    │    250ms 后读 __playCount   │
                    └──────────────┬──────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
       playCount 增加                          playCount 没变
                │                                     │
                ▼                                     ▼
   本次按键触发了新音频                    当前 slide 的 v-click 已点完
                │                                     │
                ▼                                     ▼
   waitForFunction(                       按 ArrowRight
     __lastAudio.ended === true                 │
   )   (timeout 60s)                    ┌───────┴────────┐
                │                       │                │
                ▼                  URL 变了          URL 没变
          播完，继续按 Space          │                │
                                       ▼                ▼
                                   新一页，循环       末页，结束
```

`__playCount` 是脚本自己在 `PatchedAudio` 构造时 `+1` 的全局计数器，相当于"创建了多少次 audio 元素"。从外部可以无侵入地观察到"是否真的有新 audio 被创建"。

安全阀：
- 单次音频等待 60s 超时 → 继续（防个别 click 死循环）
- 总点击 200 次或 10 分钟 → 强制结束

---

## 4. 音画对齐：性能时间戳 + 静音填充

**问题**：MediaRecorder 在 *首次 `audio.play()`* 时才有 PCM 数据，但 Playwright 录视频从 *context 创建* 那一刻就开始。两者起点天然错位。

**方案**：用 `performance.now()` 在页面内算出偏移，ffmpeg 给音频前垫相同长度的静音。

### 4.1 测量偏移

页面内（`INIT_SCRIPT`）：

```js
window.__recorderStart = performance.now();  // MediaRecorder 启动时刻
window.__firstPlayTime = null;
// PatchedAudio 内：
if (window.__firstPlayTime === null) {
  window.__firstPlayTime = performance.now() - window.__recorderStart;
}
```

Node 端收尾时把这个值回传，单位毫秒。

### 4.2 垫静音

```bash
# 偏移 < 200ms 直接转封装
ffmpeg -i narration.webm -c:a aac -b:a 192k narration-padded.m4a

# 偏移 ≥ 200ms 垫静音
ffmpeg \
  -f lavfi -t 3.842 -i anullsrc=channel_layout=stereo:sample_rate=48000 \
  -i narration.webm \
  -filter_complex '[0:a]aresample=48000,...[s];[1:a]aresample=48000,...[n];[s][n]concat=n=2:v=0:a=1[a]' \
  -map '[a]' -c:a aac -b:a 192k narration-padded.m4a
```

> 用 `performance.now()` 而不是 `Date.now()` —— 前者单调递增，不受多 tab / 系统挂起漂移影响。

### 4.3 为什么必须先转 aac m4a

webm 容器规范只允许 **VP8/VP9/AV1 视频 + Vorbis/Opus 音频**。如果直接拿 opus 装进 mp4，ffmpeg 在 mux 阶段会报：

```
Only VP8/VP9/AV1 video and Vorbis/Opus audio and WebVTT subtitles are supported for WebM
```

所以先把 webm/opus 拆出来转封装成 m4a/aac，再喂给最终的 mp4 mux。

---

## 5. 最终 mux

```bash
ffmpeg \
  -i raw.webm -i narration-padded.m4a \
  -map 0:v:0 -map 1:a:0 \
  -c:v libx264 -pix_fmt yuv420p -crf 23 -preset medium \
  -c:a aac -b:a 192k -ar 48000 \
  -movflags +faststart -shortest \
  output/slide-1.mp4
```

参数含义：

| 参数 | 作用 |
|---|---|
| `-pix_fmt yuv420p` | 兼容 QuickTime、微信、剪映等所有播放器 |
| `-crf 23 -preset medium` | 视觉无损与编码速度的折中 |
| `-movflags +faststart` | moov 原子前置，浏览器边下边播 |
| `-shortest` | 跟音轨同步结束，避免尾部空白 |

### 中间产物清单（`build/` 下）

| 文件 | 来源 |
|---|---|
| `raw.webm` | Playwright 录的视频（vp8/vp9，无音轨） |
| `narration.webm` | 页面内 MediaRecorder 抓的音频（opus） |
| `narration-padded.m4a` | 音频前垫静音 + 转 aac |
| `output/<name>.mp4` | 最终产物 |

---

## 6. 关键设计取舍

| 取舍 | 选择 | 原因 |
|---|---|---|
| 视频分辨率 | 默认 720p，可 `--width/--height` 覆盖 | 1080p 编码慢、文件大；720p 演示足够清晰 |
| v-click 数量 | 动态探测（`__playCount` 增量 + URL 变化） | 不写死数量，slides.md 改了不用改脚本 |
| 翻页键 | Space 触发 v-click，ArrowRight 切页 | 与 Slidev 内置快捷键一致 |
| 音频等待 | `__lastAudio.ended === true` | 比 `setTimeout` 按预估时长等更准，旁白可长可短 |
| 同步基准 | `performance.now()` | 单调时钟，不受系统时间调整 / 挂起影响 |
| dev server 探测 | `localhost` → `127.0.0.1` → `[::1]` 轮询 | 绕开 macOS IPv6/IPv4 解析顺序导致 fetch 失败 |

---

## 7. 关键坑（脚本里已规避）

1. **全局层 `useSlideContext().$clicks` 是死的** —— 不影响本脚本（脚本直接读自己注入的 `__playCount`），但其它全局组件要小心。
2. **webm 容器不支持 aac** —— 见 4.3，必须先转 m4a。
3. **录制时不要编辑 `slides.md` / `components/`** —— dev server HMR 会触发 reload 中断录音。
4. **每次入口默认清空 `build/`** —— 避免 `raw.webm` 累积导致 ffmpeg 多输入冲突；想调试加 `--no-clean`。
5. **headless 下音频空** —— 这是 Chromium 平台限制，不是脚本 bug。
6. **分辨率非偶数会触发 yuv420p 警告** —— 默认 1280×720 / 1920×1080 都对齐 16 像素，安全。

---

## 8. 调试技巧

```bash
# 保留 dev server 录完继续跑，方便手动看页面
pnpm record --keep-server

# 复用上一次录的 raw.webm / narration.webm，只重跑 ffmpeg 部分
pnpm record --no-clean

# 看内部状态（playCount、URL、__lastAudio.ended、__firstPlayTime 等）
DEBUG=1 pnpm record
```

录完想从视频里把开头的白屏/启动对齐段裁掉：

```bash
ffmpeg -ss 10 -i output/slide-1.mp4 -c copy output/slide-1-trimmed.mp4
```

`-ss 10 -c copy` 是按关键帧的快速裁剪，秒级完成；H.264 + AAC 流复制安全，无需重编码。精度不够时把 `-ss` 放到 `-i` 之后并去掉 `-c copy`，触发解码换取帧精度。
