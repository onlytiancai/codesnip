# 英语口语视频生成器 · 进度日志

> **完整设计文档**：[video-generator-plan.md](./video-generator-plan.md)
> **工作目录**：`scripts/video/`
> **最后更新**：2026-06-27

## 当前状态

**已完成 10 步 + 一轮 UI 升级**：Remotion demo → TTS+字幕 → Header/Footer/AI插画 → desc JSON 本地生成 → HTML 卡片预览 → 卡片音频生成+duration 回填 → 全量资产生成 → 三种卡片 Remotion 组件 → Video.tsx 端到端渲染 → **`scripts/render.ts` 分阶段 CLI** → **UI 升级（蛙蛙英语口语 + DAY N + i/M 进度）**。

当前能一条命令 `tsx scripts/render.ts scripts/output/1.json` 端到端生成 mp4，或单独跑 `desc / assets / audio / video` 任一阶段；渲染后可加 `--clean` 自动清理中间产物。视频时长 64-89s，6.3-9.5 MB，1080×1920，封面色 = 浅薄荷绿；Header 显示"蛙蛙英语口语"，IntroCard 显示 DAY N，ExpressionCard 右上角显示 i/M 进度。

---

## 下一步建议

- **Step 11**：批量跑 49 个 desc → 49 个 mp4（render CLI 已就绪，循环 + 异常处理即可）
- **Step 12**（可选）：LLM 优化文案（人话化 TTS 文案 / 增加更多 expressions）
- **Step 13**（可选）：封面图 prompt 优化 / 添加动态元素 / 字体样式精修

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

### ✅ Step 7：全量资产生成（卡片音频 + 场景插画）

**目标**：一条命令搞定所有卡片音频 + 场景插画。

**新增文件**：
- `scripts/generate-all.ts` — 复用 `synthesize()` + `generate()`，生成所有 audio + image + 回填 desc

**关键设计**：
- 复用 Step 3 的 `synthesize()` 和 `generate()`，零新代码
- `--force` 强制重生成（默认跳过已存在）
- 场景插画 prompt 派生：scene_en + task_en → `buildScenePrompt()`
- 回写：每个 segment 的 audio_path/duration_ms + 每张 card 的 duration_sec + scene_image.{prompt,url,local_path} + 顶层 duration_sec/frames

**7 张卡片总时长**：61s · 1830 帧

**关键迭代**：card 0 音频太短（仅 1.8s 太突兀）→ 把 1.json 的 `context` 加进 tts_segments 文本 → 9.33s · duration_sec=11

**关键迭代**：插画 prompt 加入 `Scene situation: ${context.split(/[，。]/)[0]}` 一句，让 AI 画图时考虑场景背景

---

### ✅ Step 9：Video.tsx 主组合（端到端渲染）

**目标**：把 desc JSON 喂给 Remotion，渲染整段视频。

**新增文件**：
- `src/Video.tsx` — 主组合，`<Sequence>` 拼接所有卡片 + 全局 Header/Footer
- `scripts/update-durations.ts` — 用新公式（含段间停顿）重算 duration_sec

**Step 9 → Step 9.1 三轮迭代**：

**v1（初次）**：bg 图 + overlay + 卡片 solid bg
- ❌ bg 图看不见：每张卡 `background: c.bg` 把全局 bg 挡死
- 修复：去掉卡片的 `background: c.bg`，让 Video 的 bg+overlay 透过来

**v2（去 bg）**：bg 图能显示但偏厚（overlay 用 0.9 alpha 太厚）
- 修复：overlay 改 0.2-0.25 alpha

**v3（用户反馈三大问题）**：
1. ❌ bg 图喧宾夺主：移到 IntroCard 内作为封面主题图，Video 不再有 bg 图层
2. ❌ card 0 渐渐出现：去掉入场动画（entrance fadeIn + translateY），第 0 帧直接显示 = 封面
3. ❌ 中英文一起读（音频重叠 bug）：
   - 原因：`<Audio>` 默认从父组件 frame 0 开始播，多个 Audio 同时播放
   - 修复：每段 Audio 包 `<Sequence from={startFrame} durationInFrames={durFrames}>` 串行播放
   - 段间停顿 PAUSE_MS=700：zh → pause → en
   - en/phonetic 淡入起始帧 = zh 段结束 + pause

**v4（duration 重算）**：
- 公式：`max(4, ceil((sum_seg_ms + (n-1)*PAUSE_MS) / 1000) + 1)`
- card 0/6（单段）duration 不变；card 1-5（双段）增加 1s
- 同步更新 `generate-card-audio.ts` 和 `generate-all.ts` 的公式

**最终结果**：
- 视频时长：65s · 1950 帧 · 6.3 MB
- 验证帧：
  - frame 0：cover 图（购物插画 + SCENE/购物 + TASK/礼品包装 + 一句话）✓
  - frame 410（zh 段播完+停顿中）：只显示 zh ✓
  - frame 460（en 段播完时）：zh + en + 音标 + note 全部显示 ✓

**踩坑记录**：
- ⚠️ `<Audio>` 默认从父 frame 0 播放，多个 Audio 同时播 → 一定要包 `<Sequence>` 或用 `startFrom` prop
- ⚠️ 卡片的 outer AbsoluteFill 不能有 `background: c.bg`，会挡死全局 bg 图
- ⚠️ 卡片 Duration 必须考虑段间停顿，否则 en 段会被截断

---

### ✅ Step 8：三种卡片 Remotion 组件

**目标**：把 Step 5 的 HTML 预览布局直接复刻为 Remotion 组件，能用 `<Audio>` 真实播音 + 帧级动效。

**新增文件**：
- `src/theme.ts` — 共享常量（LAYOUT / THEME_COLORS / STYLE_COLORS / FONT_FAMILY / `toStaticFile()` 工具）
- `src/components/cards/IntroCard.tsx` — SCENE → 大标题 → 分隔条 → TASK → task 标题 → 一句话（6 元素 stagger fade-in）
- `src/components/cards/ExpressionCard.tsx` — style badge → 中文 → 英文 → 音标 → note（5 元素，en/phonetic 在 zh 段播完后才淡入）
- `src/components/cards/SummaryCard.tsx` — eyebrow + 整段 explanation（按长度自适应字号 64/56/48/42px）

**临时回归测试**：
- `src/Card1Test.tsx` — 硬编码 desc 1.draft.json 的 card 1 数据
- `src/Root.tsx` 注册 `Card1Test` Composition（180 帧）
- 渲染 `out/card1-test.mp4`（635 KB）+ 抽 frame 108 截图验证

**关键设计**：
- **不渲染 Header/Footer**：这俩由 Step 9 的 Video.tsx 全局套，卡片组件不管
- **Audio 偏移**：每个 `<Audio>` 放在 card 自己的 AbsoluteFill 里，按 card 局部 frame 0 开始播放。父组件用 `<Sequence from={startFrame} durationInFrames={...}>` 包裹即可自动偏移
- **音频路径转换**：`audio_path` 是绝对路径，`toStaticFile()` 提取 `audio/...` 相对段给 `staticFile()`
- **en 段出现时机**：zhEndFrame = `Math.round(zh.duration_ms/1000 * fps)`，en 的 fadeIn 起始帧 = zhEndFrame
- **自适应 Summary 字号**：explanation < 200 字 → 64px，< 400 → 56px，< 600 → 48px，≥ 600 → 42px

**回归测试截图**（card 1 frame 108）：
- POLITE · 礼貌 徽章 ✓
- 中文 "您能把这个礼品包装一/下吗？" 2 行换行 ✓
- 英文 "Could you gift wrap this,/please?" 2 行换行 ✓
- 音标 "/kuːd juː ɡɪft ræp ðɪs pliːz/" 斜体 ✓
- Note 带绿色左色条 ✓

**Step 8 typecheck**：✅ `tsc --noEmit` 无错误
**Step 8 渲染回归**：✅ card1-test.mp4 6s 635 KB，布局与 HTML 预览一致

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
│   ├── Root.tsx                  # Compositions: HelloWorld + Card1Test, 各 180 帧
│   ├── HelloWorld.tsx            # 主组件（标题 + 字幕 + 音频 + 背景图 + Header/Footer）
│   ├── Card1Test.tsx             # 🆕 Step 8 回归测试：硬编码 card 1 数据
│   │
│   ├── components/
│   │   ├── Header.tsx            # 顶部条
│   │   ├── Footer.tsx            # 底部条
│   │   └── cards/                # 🆕 三种卡片组件
│   │       ├── IntroCard.tsx     # 场景引入卡（SCENE/TASK/sentence）
│   │       ├── ExpressionCard.tsx# 表达卡（badge/zh/en/phonetic/note）
│   │       └── SummaryCard.tsx   # 总结卡（eyebrow + explanation）
│   │
│   ├── api/
│   │   ├── minimax-tts.ts        # TTS 客户端（zh + en 双 voice）
│   │   └── minimax-image.ts      # text_to_image 客户端（base64 模式）
│   │
│   └── theme.ts                  # 🆕 全局主题常量（LAYOUT/THEME_COLORS/STYLE_COLORS/toStaticFile）
│
├── scripts/
│   ├── generate-assets.ts        # 旧：Hello 资源（audio + image）
│   ├── generate-desc.ts          # 🆕 纯本地 desc JSON 生成
│   ├── preview-card.ts           # 🆕 HTML 卡片预览
│   ├── generate-card-audio.ts    # 🆕 TTS 生成 + duration 回填（单 card）
│   └── generate-all.ts           # 🆕 全量资产生成（所有 audio + scene image）
│
├── public/
│   ├── audio/
│   │   ├── hello.mp3             # 91.7 KB, 5.76s, female-shaonv（demo 用）
│   │   └── 1/                    # 🆕 desc/1.draft.json 对应音频
│   │       ├── 0-0-zh.mp3        # 9.33s（context 已加入）
│   │       ├── 1-0-zh.mp3        # 2.33s
│   │       ├── 1-1-en.mp3        # 2.20s
│   │       ├── 2-0-zh.mp3        # 2.02s
│   │       ├── 2-1-en.mp3        # 1.55s
│   │       ├── 3-0-zh.mp3        # 2.05s
│   │       ├── 3-1-en.mp3        # 1.40s
│   │       ├── 4-0-zh.mp3        # 2.34s
│   │       ├── 4-1-en.mp3        # 1.80s
│   │       ├── 5-0-zh.mp3        # 2.02s
│   │       ├── 5-1-en.mp3        # 1.76s
│   │       └── 6-0-zh.mp3        # 21.82s
│   └── images/
│       ├── scene.jpg             # 58.4 KB, 720x1280, image-01 + base64（demo 用）
│       └── 1.jpg                 # 🆕 场景插画（含 Scene situation 提示）
│
├── scripts/desc/                 # 🆕 视频描述 JSON（人工审核中间产物）
│   └── 1.draft.json              # 7 张卡片，total 61s · 1830 帧
│
├── scripts/preview/              # 🆕 HTML 卡片预览
│   └── 1-card-1.html             # 5.3 KB
│
└── out/
    ├── hello.mp4                 # 656.9 KB, 1080x1920, 6s, h264+aac
    ├── card1-test.mp4            # 🆕 Step 8 回归渲染
    └── card1-frame108.png        # 🆕 Step 8 截图验收
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
| 全量资产生成（一键 audio + image） | `tsx generate-all.ts` | ✅ 12 音频 + 1 插画，61s 总时长 |
| 三种卡片 Remotion 组件 | `pnpm typecheck` + Card1Test render | ✅ card1-test.mp4 6s 635 KB |
| ExpressionCard 布局与 HTML 一致 | ffmpeg 抽 frame 108 截图 | ✅ 5 元素排版正确 |

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

---

### ✅ Step 10：render.ts 分阶段 CLI + UI 升级

**目标**：把"四步产物生成 + Remotion 渲染"打包成一条命令，且支持分阶段、强制、清理。

**新增文件**：
- `scripts/render.ts` — 端到端视频生成流水线 CLI

**关键设计**：
- **4 阶段 + 1 all**：`--phase=desc|assets|audio|video|all`，默认 `all`
- **路径基准分离**：`PROJECT_ROOT = en-sentence-study/`（input 路径基准）vs `ROOT = scripts/video/`（产物路径基准）。这避免了用户 `scripts/output/N.json` 的相对路径在不同 cwd 下解析错乱
- **复用 + 串行**：desc 阶段 spawn 到原有 `generate-desc.ts`；assets/audio/video 阶段直接 import minimax-tts/minimax-image API 调用，零重复逻辑
- **--clean 清理**：渲染完成后删除 `public/images/N.jpg`、`public/audio/N/`、`scripts/desc/N.draft.json`，仅保留最终 mp4；支持 `--keep-images/--keep-audio/--keep-desc` 部分保留
- **--force 强制**：跳过已存在检测，全部重生成
- **自动推导 desc 路径**：当只给 `--input` 而非 `--phase=desc` 时，自动从 input 文件名推导 desc 路径

**v1 → v2 迭代**：

**v1**：input path 用 `resolve(opts.input)`，相对 ROOT（video）解析
- ❌ 用户给 `scripts/output/2.json` → 解析为 `video/scripts/output/2.json`（错），找不到
- 修复：`resolve(PROJECT_ROOT, opts.input)` → 用户给相对项目根的路径

**v2**：PROJECT_ROOT 计算错误
- ❌ `resolve(ROOT, '..')` → `scripts/`（错），input 变 `scripts/scripts/output/2.json`
- 修复：`resolve(ROOT, '../..')` → `en-sentence-study/`

**v3**：input 路径修复后，desc/2.draft.json 已存在 → phaseDesc 跳过 → 端到端成功（87s · 9.5 MB · 2670 帧）

**验证**：
```bash
pnpm exec tsx scripts/render.ts scripts/output/1.json              # 端到端
pnpm exec tsx scripts/render.ts scripts/output/1.json --phase assets --force  # 仅重新生成 1.jpg
pnpm exec tsx scripts/render.ts scripts/output/1.json --phase video --clean   # 渲染 + 清理
```

**Step 10 typecheck**：✅ `tsc --noEmit` 无错误

---

### ✅ Step 10.1：UI 升级（蛙蛙英语口语 + DAY N + i/M 进度）

**目标**：让视频更有"学习产品"的感觉：标识品牌、明确第几天、显示进度。

**改动文件**：
| 文件 | 改动 |
|------|------|
| `src/Video.tsx` | Header 默认 `英语口语·每日一句` → `蛙蛙英语口语`；计算 `dayNumber`（从 `desc.id` 末尾数字提取）和 `exprIdx/total`（expression card 进度） |
| `src/components/cards/IntroCard.tsx` | 新增 `dayNumber: number` prop；在 SCENE 上方加绿色 "DAY N" 胶囊徽章 |
| `src/components/cards/ExpressionCard.tsx` | 新增 `progress?: {current,total}` prop；右上角加绿色 "i/M" 胶囊（仅 expression card） |

**关键设计**：
- **Day N 提取**：`parseInt(desc.id.match(/(\d+)$/)?.[1] ?? '0', 10) || 1`，兼容 `1.json` / `49.json` / `draft-1.json` 等命名
- **进度 i/M**：分子 = 当前 card 在 expression 列表中的位置（1-based），分母 = 所有 expression cards 总数
- **胶囊样式**：圆角 100px，绿色背景 `c.accent`，白色文字 + 阴影；与 POLITE/NEUTRAL/CASUAL/BOLD 风格徽章视觉一致

**验证（frame 抽帧）**：
- `out/v3-cover-1.png`（1.json frame 0）：DAY 1 徽章 + SCENE/购物 + TASK/礼品包装 ✓
- `out/v3-card1.png`（1.json frame 500）：Header "蛙蛙英语口语" + 右上角 "1/5" + POLITE 徽章 + 中文 + 英文 + 音标 + Note ✓
- `out/v3-card3.png`（1.json frame 900）：右上角 "4/5"（i 动态变化）✓
- `out/v3-cover-2.png`（2.json frame 0）：DAY 2 徽章 ✓

**额外修复（图片裁剪）**：
- 用户报告横版插图右侧被裁剪
- 原代码：`width: 'auto'` + `left:60, right:60` + `height:760` + `objectFit: cover`
  - 1280×720 图按 height=760 缩放后 width=1351px，超出 1080 画布
- 改为：`width: 1080 × 0.9 = 972px`（宽 90%）+ `left: 54px`（水平居中）+ `height: 547px`（按 16:9 自适应）
  - 完整显示无裁剪
- 文字区起始位置下移：`textTop = imageTop + imageHeight + 40 = 737px`（原 950px）

---

### ✅ Step 10.2：图片裁剪问题最终修复

**问题**：1280×720 横版插图按 height=760 缩放后 width=1351px，超出 1080 画布导致右侧被裁剪。

**原代码**（`IntroCard.tsx`）：
```tsx
<Img
  style={{
    top: imageTop, left: 60, right: 60,
    height: imageHeight,        // 760
    width: 'auto',              // ← 罪魁！让 width 跟 height 算成 1351px
    objectFit: 'cover',
  }}
/>
```

**修复**（`IntroCard.tsx`）：
```tsx
const imageWidth = 1080 * 0.9;             // 972px（90% 宽）
const imageLeft = (1080 - imageWidth) / 2;  // 54px 居中
const imageHeight = imageWidth / (16 / 9);  // 547px（16:9 自适应）

<Img
  style={{
    top: imageTop, left: imageLeft,
    width: imageWidth, height: imageHeight,
    // 移除 width: 'auto' 和 objectFit: cover（精确尺寸无需裁切）
    borderRadius: 32,
    boxShadow: '0 12px 40px rgba(14, 59, 46, 0.18)',
  }}
/>
```

**效果**：插图完整显示，左右两侧不再被裁剪；下方文字区起始位置从 950px 下移到 737px，给文字更多呼吸空间。

---

## 当前文件结构

```
scripts/video/
├── package.json                  # pnpm@10.26.0, type:module
├── tsconfig.json                 # ES2022 + react-jsx
├── remotion.config.ts            # jpeg 帧格式, overwrite, concurrency=1
│
├── src/
│   ├── index.ts                  # registerRoot
│   ├── Root.tsx                  # Compositions: HelloWorld / Card1Test / EnSentenceVideo
│   ├── Video.tsx                 # 🆕 Step 9 主组合：Sequence 拼接 + 全局 Header/Footer
│   ├── HelloWorld.tsx            # demo
│   ├── Card1Test.tsx             # Step 8 回归测试
│   │
│   ├── components/
│   │   ├── Header.tsx            # 顶部条（Step 10.1 文案改"蛙蛙英语口语"）
│   │   ├── Footer.tsx            # 底部条
│   │   └── cards/
│   │       ├── IntroCard.tsx     # 🆕 Step 10.1 加 DAY N 徽章；Step 10.2 图片改 90% 宽自适应
│   │       ├── ExpressionCard.tsx# 🆕 Step 10.1 加 i/M 进度；Step 9 zh→pause→en 音频错位修复
│   │       └── SummaryCard.tsx   # 总结卡（eyebrow + explanation 自适应字号）
│   │
│   ├── api/
│   │   ├── minimax-tts.ts        # TTS 客户端（zh + en 双 voice）
│   │   └── minimax-image.ts      # text_to_image 客户端（base64 模式）
│   │
│   └── theme.ts                  # 全局主题常量（LAYOUT/THEME_COLORS/STYLE_COLORS/toStaticFile）
│
├── scripts/
│   ├── generate-assets.ts        # 旧：Hello 资源（audio + image）
│   ├── generate-desc.ts          # 纯本地 desc JSON 生成
│   ├── preview-card.ts           # HTML 卡片预览
│   ├── generate-card-audio.ts    # TTS 生成 + duration 回填（单 card）
│   ├── generate-all.ts           # 全量资产生成（所有 audio + scene image）
│   └── render.ts                 # 🆕 Step 10 端到端 CLI（desc/assets/audio/video 分阶段 + --clean）
│
├── public/
│   ├── audio/
│   │   ├── hello.mp3             # demo
│   │   └── N/                    # desc N 对应音频（按 cardIdx-segIdx-lang 命名）
│   └── images/
│       ├── scene.jpg             # demo
│       └── N.jpg                 # 🆕 场景插画（1280x720 横版，含 Scene situation 提示）
│
├── scripts/desc/                 # 视频描述 JSON（人工审核中间产物）
├── scripts/preview/              # HTML 卡片预览
└── out/                          # 渲染输出（mp4 + 帧截图）
    ├── 1-desc.mp4                # 🆕 Step 10 端到端渲染（64s · 6.3 MB）
    └── 2-desc.mp4                # 🆕 Step 10 端到端渲染（89s · 9.5 MB）
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
| 全量资产生成（一键 audio + image） | `tsx generate-all.ts` | ✅ 12 音频 + 1 插画，61s 总时长 |
| 三种卡片 Remotion 组件 | `pnpm typecheck` + Card1Test render | ✅ card1-test.mp4 6s 635 KB |
| ExpressionCard 布局与 HTML 一致 | ffmpeg 抽 frame 108 截图 | ✅ 5 元素排版正确 |
| Video.tsx 端到端拼接 | `remotion render ... --props` | ✅ 1-desc.mp4 65s · 6.3 MB |
| `<Audio>` 错位修复（zh → pause → en） | ffmpeg 抽多帧 + 听音 | ✅ zh 段播完停顿后再播 en |
| Remotion `calculateMetadata` | desc 改 fps/duration 后重渲染 | ✅ 动态 durationInFrames |
| `scripts/render.ts` 端到端 CLI | 跑 1.json/2.json | ✅ 64s/89s mp4 |
| `scripts/render.ts` `--phase` 分阶段 | 单独跑 assets/video | ✅ 复用 + 增量 |
| `scripts/render.ts` `--clean` 清理 | 渲染后删除中间产物 | ✅ 仅保留 mp4 |
| IntroCard 90% 宽 + 自适应高（不裁剪） | ffmpeg 抽 frame 0 | ✅ 1280×720 图完整显示 |
| DAY N 徽章（从 desc.id 提取） | ffmpeg 抽 cover 帧 | ✅ DAY 1 / DAY 2 |
| i/M 进度（仅 expression card） | ffmpeg 抽多张 expression card | ✅ 1/5 → 4/5 动态 |

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
10. **`<Audio>` 默认从父 frame 0 播放**：多个 Audio 同时播，必须包 `<Sequence from={...}>` 串行
11. **卡片 outer AbsoluteFill 不能有 `background: c.bg`**：会挡死全局 bg 图
12. **横版图 + `width: 'auto'` 会溢出**：1280×720 按 height=760 缩放后 width=1351px，超出 1080 画布。改用精确 `width` + 自适应 `height`（或反之）
13. **`resolve()` 的基准是当前 cwd**：跨目录 CLI 的相对路径要显式 `resolve(PROJECT_ROOT, input)`，否则会因 cwd 不同解析错乱
14. **PROJECT_ROOT 容易算错**：要搞清楚是几层 `..`，测试时直接打印 `path.resolve(ROOT, '../..')` 确认
15. **render.ts 用 `spawnSync` 调用已有脚本**：如果直接 `import` 主函数，会重复执行一遍；如果只想"调用 + 传参"，`spawnSync` 更省事 |