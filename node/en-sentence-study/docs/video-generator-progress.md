# 英语口语视频生成器 · 进度日志

> **完整设计文档**：[video-generator-plan.md](./video-generator-plan.md)
> **工作目录**：`scripts/video/`
> **最后更新**：2026-06-27

## 当前状态

**已完成 6 步**：Remotion demo → TTS+字幕 → Header/Footer/AI插画 → desc JSON 本地生成 → HTML 卡片预览 → 卡片音频生成+duration 回填。

当前能**纯本地**把 `scripts/output/N.json` 转成 `scripts/desc/N.draft.json`，用 HTML 预览卡片排版，再用 TTS 生成各卡片的音频（zh+en 双音色）并自动回填 duration。**距离渲染真实场景视频只差一步**：把 desc JSON 喂给 Remotion。

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

### ✅ Step 4：desc JSON 本地生成脚本

**目标**：从 `scripts/output/N.json` 纯本地生成 `scripts/desc/N.draft.json`，作为后续所有步骤的输入契约。**不调用任何外部 API**。

**关键设计决策**（用户提的）：
- `duration_sec` 一律 `-1`（占位，待音频生成后回填）
- **card 0 (intro)**：`tts_segments = [{lang:'zh', text: scene_zh + task_zh}]`，仅中文
- **card 1..N (expression)**：`tts_segments = [{zh, literal_translation}, {en, sentence}]`，先中后英
- **card N+1 (summary)**：`tts_segments = [{lang:'zh', text: explanation}]`，仅中文
- 字段最小化：不写 `scene_image`、不写 header/footer config、不写"人话化" TTS 文案（用源字段原文）

**新增文件**：
- `scripts/generate-desc.ts` — 纯本地 desc 生成器（带源 JSON schema 校验）
- `scripts/desc/1.draft.json` — 重生成（替换了之前手写的版本）

**验证**：
- 7 张卡片：`0:intro(1段) · 1-5:expression(2段) · 6:summary(1段)`
- 所有 `duration_sec = -1`

---

### ✅ Step 5：HTML 卡片预览（用 Chrome DevTools MCP 验收排版）

**目标**：用户要求"生成 card 1 的画面"用于预览排版，决定用 **HTML**（比图片快、可交互、手机框可视化）。

**新增文件**：
- `scripts/preview-card.ts` — 把指定卡片渲染成 HTML
- `scripts/preview/1-card-1.html` — card 1 预览

**HTML 设计要点**：
- **手机框**：375×812 容器，内部分辨率仍用 1080×1920 缩放 0.347×（与视频 1:1 像素）
- **5 元素排版**：POLITE 徽章 → 中文 → 英文 → 音标 → note 框（带绿色左色条）
- **CSS 自动换行**：`word-wrap: break-word; word-break: break-word;` 应用到所有文字
- **左上角 meta 面板**：显示 card 类型、duration 状态、TTS 段列表（开发调试用，正式视频不会渲染）

**关键发现** ⚠️：
- 启动 Chrome 调试端口：`/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir=/Users/huhao/chrome-profile-for-ai`
- 用 `mcp__chrome-devtools__navigate_page` + `mcp__chrome-devtools__take_screenshot` 直接截图验收，无需打开 Finder

**用户验收**：截图显示中文"您能把这个礼品包装一下吗？"自动 2 行换行，英文"Could you gift wrap this, please?"也 2 行换行，POLITE 徽章显眼，note 带绿色边框，整体观感清爽。

---

### ✅ Step 6：卡片音频生成 + duration_sec 回填

**目标**：根据 desc JSON 调 TTS 生成指定卡片的音频（中英双音色），按实际时长回填 `duration_sec`。

**新增文件**：
- `scripts/generate-card-audio.ts` — TTS 调用 + duration 计算

**duration 计算公式**：
```
duration_sec = max(4, ceil(sum(seg.duration_ms) / 1000) + 1)
```
- 至少 4 秒
- 超出则按总音频向上取整 + 1 秒留给动画
- 例：zh 2.27s + en 2.05s = 4.32s → max(4, 5+1) = **6s**

**音频文件路径**：
```
public/audio/<desc-id>/<cardIdx>-<segIdx>-<lang>.mp3
例：public/audio/1/1-0-zh.mp3 · 1-1-en.mp3
```

**关键设计**：
- **已存在音频自动跳过**：方便重跑（避免重复扣 TTS 配额）
- **支持 ffprobe 拿真实时长**：跳过时调 `ffprobe` 读时长；无 ffprobe 时按"4 字/秒"估算
- **CLI 参数**：`tsx scripts/generate-card-audio.ts <desc.json> <cardIdx|all>`，默认所有卡片

**Card 1 实际产出**：
| 段 | 文本 | 时长 | 文件 |
|----|------|------|------|
| 0-zh | 您能把这个礼品包装一下吗？ | 2.27s | `1-0-zh.mp3` (38 KB) |
| 1-en | Could you gift wrap this, please? | 2.05s | `1-1-en.mp3` (34.5 KB) |
| **合计** |  | **4.32s** | → **duration_sec = 6** |

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
│       ├── minimax-tts.ts        # TTS 客户端（zh + en 双 voice）
│       └── minimax-image.ts      # text_to_image 客户端（base64 模式）
│
├── scripts/
│   ├── generate-assets.ts        # 旧：Hello 资源（audio + image）
│   ├── generate-desc.ts          # 🆕 纯本地 desc JSON 生成
│   ├── preview-card.ts           # 🆕 HTML 卡片预览
│   └── generate-card-audio.ts    # 🆕 TTS 生成 + duration 回填
│
├── public/
│   ├── audio/
│   │   ├── hello.mp3             # 91.7 KB, 5.76s, female-shaonv（demo 用）
│   │   └── 1/                    # 🆕 desc/1.draft.json 对应音频
│   │       ├── 1-0-zh.mp3        # 38 KB, 2.27s
│   │       └── 1-1-en.mp3        # 34.5 KB, 2.05s
│   └── images/
│       └── scene.jpg             # 58.4 KB, 720x1280, image-01 + base64（demo 用）
│
├── scripts/desc/                 # 🆕 视频描述 JSON（人工审核中间产物）
│   └── 1.draft.json              # 7 张卡片，card 1 duration_sec=6（已生成音频）
│
├── scripts/preview/              # 🆕 HTML 卡片预览
│   └── 1-card-1.html             # 5.3 KB
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
| Chrome DevTools MCP + 调试端口 | `--remote-debugging-port=9222` | ✅ 截图验收 HTML 预览 |
| 纯本地 desc JSON 生成（无 API） | `tsx generate-desc.ts` | ✅ 7 张卡片 |
| HTML 卡片预览（手机框 + 自动换行） | Chrome screenshot | ✅ card 1 排版清晰 |
| TTS 双音色（zh + en）+ duration 回填 | `tsx generate-card-audio.ts` | ✅ card 1 duration_sec=6 |

---

## 已踩过的坑（避坑参考）

1. **`@remotion/cli` 必须显式 install**：Remotion 主包不含 CLI
2. **esbuild postinstall 被 pnpm 默认忽略**：需要 `pnpm rebuild esbuild`
3. **image API endpoint 不是 `/v1/text_to_image`**：官方文档无对应路径，正确是 `/v1/image_generation`
4. **签名 URL 时钟漂移**：用 base64 模式，不要默认 url 模式
5. **图片太小看不出来**：`imageOpacity × overlayOpacity × blur` 三者叠加，初始值不能小于 0.5 才看得清
6. **视频时长 ≠ 音频时长**：先调 TTS 拿到 `audio_length` 再决定 `durationInFrames`
7. **desc JSON 字段最小化**：用户明确说"不必要的字段可以去掉"——LLM 还没接入，scene_image.prompt 也暂未生成（需要 text_to_image API）；先聚焦纯本地流程
8. **duration_sec 必须等音频**：先填 `-1` 占位，TTS 后再回填，不要写死 6 秒这种魔法数字
9. **Chrome DevTools MCP 需要 Chrome 启动带调试端口**：CLAUDE.md 提供了启动命令

---

## 下一步建议

按计划推进：

- **Step 7**（最近）：把所有 7 张卡片的音频都生成（一条命令搞定），desc 1.draft.json 填齐全部 duration
- **Step 8**：实现 IntroCard / ExpressionCard / SummaryCard 三个 Remotion 组件（基于 Step 5 的 HTML 预览布局直接复刻）
- **Step 9**：实现 `Video.tsx` 主组合（Sequence 拼接 + Header/Footer 套全局），feed desc JSON 作为 inputProps
- **Step 10**：替换 HelloWorld 为 EnSentenceVideo Composition，跑通 `scripts/output/1.json` 端到端
- **Step 11**：批量处理 49 个 JSON（人工审核中间产物，可选 LLM 优化文案）

---

## 可复用资产

如果想换一个 demo 主题，只需要改：

- `src/HelloWorld.tsx` 中的 `THEME` 变量（`'mint' | 'sunny'`）
- `scripts/generate-assets.ts` 中的 `IMAGE_PROMPT` 和 `TTS_TEXT`
- `src/components/Header.tsx` 和 `Footer.tsx` 顶部的 `COLORS` 对象
- `scripts/preview-card.ts` 顶部的 `STYLE_COLORS` 表（POLITE/NEUTRAL/CASUAL/BOLD 配色）

不需要改任何 API 调用代码。

---

## 设计决策记录（Step 4–6 用户提的关键约束）

| 约束 | 落实位置 |
|------|----------|
| duration_sec 不要写死 | `generate-desc.ts` 输出 `-1`，`generate-card-audio.ts` 回填 |
| 双音色（zh + en） | `tts_segments: [{lang:'zh', text}, {lang:'en', text}]`；TTS 调用时按 lang 选 voice |
| card 0 只读中文 | `tts_segments: [{lang:'zh', text: scene_zh + task_zh}]` |
| 画面要显示 中文/英文/音标/note/style | `preview-card.ts` 中 `.zh-line / .en-line / .phonetic / .note / .style-badge` 五大元素 |
| 自动换行 | CSS `word-wrap: break-word; word-break: break-word;` 应用到所有文字元素 |
| 不需要人话化 | `generate-desc.ts` 直接用源字段原文，不调 LLM |
| 纯本地生成 | `generate-desc.ts` 0 个外部依赖、0 次 API 调用 |