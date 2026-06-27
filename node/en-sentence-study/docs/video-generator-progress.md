# 英语口语视频生成器 · 进度日志

> **完整设计文档**：[video-generator-plan.md](./video-generator-plan.md)
> **工作目录**：`scripts/video/`
> **最后更新**：2026-06-27

## 当前状态

**已完成 4 步**，技术栈全部跑通。当前能基于 `scripts/output/N.json`（源数据）+ AI 生成内容 + TTS + AI 插画，输出一段 1080×1920、6 秒、带 Header/Footer/中英字幕的 demo 视频。

下一个里程碑：把 demo 抽象成 **视频描述 JSON**（zod schema），让 LLM 自动把源 JSON 转成结构化描述，再喂给 Remotion 渲染。

---

## 已完成步骤

### ✅ Step 1：Remotion 环境搭建 + Hello-World 视频

**目标**：验证 macOS arm64 + Remotion 4 跑得通。

**关键产物**：
- `scripts/video/package.json` — pnpm 工程
- `scripts/video/tsconfig.json` — TS 配置
- `scripts/video/remotion.config.ts` — Remotion CLI 配置
- `src/index.ts` — registerRoot 入口
- `src/Root.tsx` — Composition 注册（150 帧 5 秒）
- `src/HelloWorld.tsx` — 简单视频内容

**渲染结果**：1080×1920 H.264，5 秒，597 KB。

---

### ✅ Step 2：TTS 语音 + 字幕

**目标**：验证 MiniMax TTS API + 字幕同步。

**关键改动**：
- 写 `scripts/tts-hello.mjs` 调用 MiniMax `/v1/t2a_v2`
- 验证：返回 hex 编码音频，5.72 秒中文女声 `female-shaonv`
- `src/HelloWorld.tsx` 加入 4 段字幕 + `<Audio>` 组件
- 视频延长到 180 帧（6 秒）以容纳音频

**字幕与 TTS 文本对应**：
| 帧 | 时间 | 字幕 |
|----|------|------|
| 0–21 | 0.0–0.7s | Hello, |
| 21–66 | 0.7–2.2s | 欢迎使用 Remotion |
| 66–111 | 2.2–3.7s | 英语口语视频生成器 |
| 111–171 | 3.7–5.7s | 环境测试运行成功 |

**音频**：AAC 立体声 48kHz，5.99 秒。

---

### ✅ Step 3：Header/Footer + TTS 模块抽取 + AI 插画

**目标**：把临时脚本升级成正式模块，加 Header/Footer，加 AI 插画作为背景。

**新增模块**：
- `src/api/minimax-tts.ts` — 正式 TS 模块，支持 zh/en 双 voice_id、hex/base64 嗅探
- `src/api/minimax-image.ts` — text_to_image 封装，支持 url/base64 双模式

**新增组件**：
- `src/components/Header.tsx` — 顶部条，spring 滑入，薄荷绿主题色
- `src/components/Footer.tsx` — 底部条，淡入 + 上移，半透明 + 毛玻璃

**编排脚本**：
- `scripts/generate-assets.ts` — 用 `tsx` 一键生成音频 + 图片（替代旧 `tts-hello.mjs`，已删除）

**关键发现** ⚠️：

1. **正确的 image API endpoint**：`POST https://api.minimaxi.com/v1/image_generation`
   - 不是 `/v1/text_to_image`（这个 404）
   - 响应字段是 `data.image_urls[]`（数组）或 `data.image_base64[]`

2. **签名 URL 403 问题**：MiniMax API 服务端时钟与 OSS 桶时钟漂移
   - 返回的 URL 永远"已过期"，无法 `fetch` 下载
   - 解决：默认改用 `response_format: 'base64'` 直存，绕过下载
   - CLAUDE.md 强调的 URL 解码（`%2F` → `/`）仍保留在 `decodeURIComponent()`，未来切回 url 模式可立即生效

---

### ✅ Step 3.5：修复插画不可见

**问题**：用户报告视频里看不到图片。

**根因**（通过 `ffmpeg` 提取第 60 帧分析）：
- 图片 opacity = 0.35（首 30 帧淡入目标值）
- 蒙版 opacity = 0.7–0.85（薄荷绿渐变）
- 叠加 + blur(4px) = 实际可见度仅 5–10%

**修复**（`src/HelloWorld.tsx`）：
- `imageOpacity`: 0 → 0.65（提到原来 2 倍）
- 蒙版：0.7–0.85 → 0.2–0.25
- `filter: blur(4px)` → `blur(2px)`

**验证**：重新提取第 60 帧，书本 + 火箭插画清晰可见。

---

## 当前文件结构

```
scripts/video/
├── package.json                  # pnpm@10.26.0, type:module
├── tsconfig.json                 # ES2022 + react-jsx
├── remotion.config.ts            # jpeg 帧格式, overwrite
│
├── src/
│   ├── index.ts                  # registerRoot
│   ├── Root.tsx                  # Composition: HelloWorld, 180 帧
│   ├── HelloWorld.tsx            # 主组件（标题 + 字幕 + 音频 + 背景图 + Header/Footer）
│   │
│   ├── components/
│   │   ├── Header.tsx            # 顶部条
│   │   └── Footer.tsx            # 底部条
│   │
│   └── api/
│       ├── minimax-tts.ts        # TTS 客户端
│       └── minimax-image.ts      # text_to_image 客户端
│
├── scripts/
│   └── generate-assets.ts        # 资产生成编排（tsx）
│
├── public/
│   ├── audio/
│   │   └── hello.mp3             # 91.7 KB, 5.76s, female-shaonv
│   └── images/
│       └── scene.jpg             # 58.4 KB, 720x1280, image-01 + base64
│
└── out/
    └── hello.mp4                 # 656.9 KB, 1080x1920, 6s, h264+aac
```

---

## 已验证的技术点

| 项 | 验证方式 | 结果 |
|----|----------|------|
| Remotion 4 在 arm64 macOS 安装 | `pnpm install` | ✅ 14.1s |
| esbuild postinstall | `pnpm rebuild esbuild` | ✅ |
| H.264 渲染 | `remotion render` | ✅ 597 KB / 6s |
| MiniMax TTS hex 返回 | curl + node | ✅ 186,600 chars → 91.1 KB MP3 |
| MiniMax TTS extra_info.audio_length | API response | ✅ 5724 ms 精确 |
| MiniMax text_to_image endpoint | curl + docs | ✅ `/v1/image_generation` |
| MiniMax base64 模式 | API call | ✅ 77,824 chars → 58.4 KB JPEG |
| MiniMax url 模式（签名 URL 下载） | curl | ❌ 403（时钟漂移） |
| `<Audio>` + `<Img>` 静态资源 | Remotion bundle | ✅ |
| Header/Footer 动画 | `spring` + `interpolate` | ✅ |
| ffmpeg ffprobe | 内置 | ✅ 校验视频流 |

---

## 已踩过的坑（避坑参考）

1. **`@remotion/cli` 必须显式 install**：Remotion 主包不含 CLI
2. **esbuild postinstall 被 pnpm 默认忽略**：需要 `pnpm rebuild esbuild`
3. **image API endpoint 不是 `/v1/text_to_image`**：官方文档无对应路径，正确是 `/v1/image_generation`
4. **签名 URL 时钟漂移**：用 base64 模式，不要默认 url 模式
5. **图片太小看不出来**：`imageOpacity × overlayOpacity × blur` 三者叠加，初始值不能小于 0.5 才看得清
6. **视频时长 ≠ 音频时长**：先调 TTS 拿到 `audio_length` 再决定 `durationInFrames`

---

## 下一步建议

按计划推进：

- **Step 4**：实现视频描述 JSON（zod schema）+ 源 JSON 解析器（schema/source.ts）
- **Step 5**：实现 IntroCard / ExpressionCard / SummaryCard 三种卡片组件
- **Step 6**：接 LLM（MiniMax Anthropic 兼容层），让 LLM 自动把源 JSON 转换成描述 JSON
- **Step 7**：实现 `video:desc` 子命令（CLI 入口）
- **Step 8**：实现 `video:render` 子命令，串联 TTS + Remotion 渲染
- **Step 9**：用 `scripts/output/1.json` 端到端跑通，输出首个真实场景视频
- **Step 10**：批量处理 49 个 JSON（人工审核中间产物）

---

## 可复用资产

如果想换一个 demo 主题，只需要改：

- `src/HelloWorld.tsx` 中的 `THEME` 变量（`'mint' | 'sunny'`）
- `scripts/generate-assets.ts` 中的 `IMAGE_PROMPT` 和 `TTS_TEXT`
- `src/components/Header.tsx` 和 `Footer.tsx` 顶部的 `COLORS` 对象

不需要改任何 API 调用代码。