# en-sentence-video · 英语口语视频生成器

把 `scripts/output/N.json`（场景+任务+多句英语表达+总结）自动转成 1080×1920 移动端英语口语讲解视频。

## 这是什么

- **输入**：`scripts/output/1.json` ~ `49.json`，每份描述一个英语场景（如购物/餐厅/机场）
- **输出**：`out/N-desc.mp4`，60~90 秒移动端竖屏视频
- **音频**：MiniMax TTS，中文女声 + 英文女声，zh → 停顿 700ms → en
- **图像**：MiniMax text_to_image 生成场景插画，作为 IntroCard 封面
- **主题**：薄荷绿（mint）/ 暖橙（sunny）两套色

## 视频样式

```
┌─────────────────────────┐
│ Header (薄荷绿, 110px)  │   全程固定
├─────────────────────────┤
│                         │
│   [场景插画 - 封面]      │   ← card 0：封面，0 帧直接显示
│                         │
│   SCENE / 购物           │
│   TASK / 礼品包装        │
│   一句话                │
│                         │
├─────────────────────────┤
│   POLITE 徽章            │   ← card 1..N：英语表达
│                         │
│   中文                   │   zh 段播放
│   英文                   │   停顿后 en 段播放
│   /音标/                 │
│   ▌ note                │
│                         │
├─────────────────────────┤
│ Footer (90px)            │
└─────────────────────────┘
```

## 工作流（两阶段）

```
scripts/output/N.json
       │
       │ ① generate-desc.ts        (纯本地，无 API)
       ▼
scripts/desc/N.draft.json           ← 人工审核中间产物
       │
       │ ② generate-all.ts          (TTS + text_to_image)
       ▼
public/audio/N/*.mp3  +  public/images/N.jpg
       │
       │ ③ remotion render          (Remotion)
       ▼
out/N-desc.mp4                      ← 端到端视频
```

**两阶段的意义**：用户可以在 desc JSON 上手改 TTS 文案、调整顺序，再渲染。

## 快速开始

```bash
cd scripts/video

# 1. 装依赖（pnpm 10+）
pnpm install && pnpm rebuild esbuild

# 2. 设置 API key（~/.zshrc 已配也行）
export MINIMAX_API_KEY=eyJhbGc...

# 3. 端到端跑一份 desc 1.json
pnpm exec tsx scripts/generate-desc.ts ../output/1.json
pnpm exec tsx scripts/generate-all.ts scripts/desc/1.draft.json
pnpm exec remotion render src/index.ts EnSentenceVideo out/1-desc.mp4

# 4. 打开视频
open out/1-desc.mp4
```

## 详细命令

| 命令 | 作用 |
|------|------|
| `pnpm exec tsx scripts/generate-desc.ts [源JSON]` | 生成 desc JSON（默认 `../output/1.json`） |
| `pnpm exec tsx scripts/preview-card.ts [desc.json] [cardIdx]` | 生成 HTML 预览（默认 card 1） |
| `pnpm exec tsx scripts/generate-card-audio.ts [desc.json] [cardIdx\|all]` | TTS 单/全部卡片音频 + 回填 duration |
| `pnpm exec tsx scripts/generate-all.ts [desc.json] [--force]` | 一键生成所有音频 + 场景插画 |
| `pnpm exec tsx scripts/update-durations.ts [desc.json]` | 用新公式（含段间停顿）重算 duration_sec |
| `pnpm exec remotion render src/index.ts EnSentenceVideo out/X.mp4` | 渲染视频（已写死的 defaultProps） |
| `pnpm exec remotion studio src/index.ts` | Remotion Studio（实时预览） |
| `pnpm typecheck` | TypeScript 类型检查 |

## 文件结构

```
scripts/video/
├── src/
│   ├── index.ts                       # registerRoot
│   ├── Root.tsx                       # Compositions 注册
│   ├── Video.tsx                      # 主组合：Sequence 拼接 + Header/Footer
│   ├── HelloWorld.tsx                 # demo（Step 1）
│   ├── Card1Test.tsx                  # Step 8 回归测试
│   ├── theme.ts                       # 全局主题常量
│   ├── components/
│   │   ├── Header.tsx                 # 顶部条
│   │   ├── Footer.tsx                 # 底部条
│   │   └── cards/
│   │       ├── IntroCard.tsx          # 封面卡（图 + SCENE/TASK）
│   │       ├── ExpressionCard.tsx     # 表达卡（badge/zh/en/phonetic/note）
│   │       └── SummaryCard.tsx        # 总结卡（eyebrow + explanation）
│   └── api/
│       ├── minimax-tts.ts             # TTS 客户端
│       └── minimax-image.ts           # text_to_image 客户端
│
├── scripts/                           # CLI 脚本
│   ├── generate-desc.ts               # 本地 desc 生成器
│   ├── preview-card.ts                # HTML 卡片预览
│   ├── generate-card-audio.ts         # TTS + duration 回填（单/全部）
│   ├── generate-all.ts                # 全量资产生成
│   └── update-durations.ts            # 用新公式重算 duration
│
├── public/
│   ├── audio/N/                       # TTS 音频（按 desc id 分目录）
│   └── images/N.jpg                   # 场景插画
│
├── scripts/desc/                      # desc JSON（人工审核中间产物）
├── scripts/preview/                   # HTML 预览
└── out/                               # 渲染输出（mp4 + 帧截图）
```

## 配置

### 主题色

`src/theme.ts`：

```ts
THEME_COLORS = {
  mint:  { bg: '#F4FBF8', text: '#0E3B2E', accent: '#19A974', ... },
  sunny: { bg: '#FFF8EB', text: '#3E2A14', accent: '#F58A1F', ... },
}

STYLE_COLORS = {
  polite:  { bg: '#A8E6CF', text: '#0E3B2E', label: 'POLITE · 礼貌' },
  neutral: { bg: '#CFE4F5', text: '#0E3B2E', label: 'NEUTRAL · 中性' },
  casual:  { bg: '#FFD6A8', text: '#3E2A14', label: 'CASUAL · 口语' },
  bold:    { bg: '#F5B7C5', text: '#3E2A14', label: 'BOLD · 直接' },
}
```

要换主题：改 `src/Root.tsx` 的 `defaultDesc.theme`，或在 desc JSON 的 `theme` 字段指定。

### TTS 语音

`src/api/minimax-tts.ts`：

```ts
TTS_VOICES = {
  zh: 'female-shaonv',
  en: 'English_PassionateWarrior',
}
```

### Header / Footer 文案

`src/Video.tsx` 里的默认值：

```tsx
<Header text="英语口语 · 每日一句" ... />
<Footer text="@en-sentence-study" ... />
```

可通过 `VideoProps.headerText` / `footerText` 覆盖（Step 10 render CLI 会暴露）。

### 段间停顿

`src/theme.ts`：

```ts
PAUSE_MS = 700;   // zh → en 之间的静音
```

要调整同时改 `generate-card-audio.ts` / `generate-all.ts` 里的同名常量。

## 渲染参数

`<Composition>`（`src/Root.tsx`）：

```tsx
durationInFrames={defaultDesc.duration_frames}   // 由 desc JSON 提供
fps={defaultDesc.fps}                            // 默认 30
width={1080}
height={1920}
```

## 已知限制 / 避坑

| 坑 | 解决方案 |
|----|---------|
| pnpm 不跑 esbuild postinstall | `pnpm rebuild esbuild` |
| MiniMax 签名 URL 时钟漂移 403 | `text_to_image` 默认 base64 模式 |
| `<Audio>` 默认从父 frame 0 播 → 多段重叠 | 每段 Audio 包 `<Sequence from={...}>` |
| 卡片 outer `background: c.bg` 挡死全局 bg | 卡片不要设 bg，让全局 bg 透过来 |
| overlay alpha > 0.5 把 bg 图挡死 | 保持 0.2-0.25 |
| duration_sec 必须含段间停顿 | 用 `update-durations.ts` 或新公式 |

## 路线图

- [x] Step 1-9：环境 → 组件 → 端到端渲染（详情见 `docs/video-generator-progress.md`）
- [ ] **Step 10**：`scripts/render.ts` CLI，输入 desc 路径 → 输出 .mp4（喂 inputProps）
- [ ] **Step 11**：批量跑 49 个 desc → 49 个 mp4
- [ ] **Step 12**：LLM 优化文案（人话化 TTS / 增加 expressions）