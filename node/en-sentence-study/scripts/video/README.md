# en-sentence-video · 蛙蛙英语口语视频生成器

把 `scripts/output/N.json`（场景+任务+多句英语表达+总结）自动转成 1080×1920 移动端英语口语讲解视频。

## 这是什么

- **输入**：`scripts/output/1.json` ~ `49.json`，每份描述一个英语场景（如购物/餐厅/机场）
- **输出**：`out/N-desc.mp4`，60~90 秒移动端竖屏视频
- **音频**：MiniMax TTS，中文女声 + 英文女声，zh → 停顿 700ms → en
- **图像**：MiniMax text_to_image 生成 16:9 横版场景插画，作为 IntroCard 封面
- **主题**：薄荷绿（mint）/ 暖橙（sunny）两套色
- **UI 元素**：Header 显示"蛙蛙英语口语"、IntroCard 显示 "DAY N"、ExpressionCard 显示 "i/M" 进度

## 视频样式

```
┌─────────────────────────┐
│ ▌ 蛙蛙英语口语           │   Header 全程固定
├─────────────────────────┤
│                         │
│   [16:9 横版插画 - 封面]  │   ← card 0：封面，0 帧直接显示
│                         │
│   [DAY 1] 徽章           │
│   SCENE / 购物           │
│   TASK / 礼品包装        │
│   一句话                │
│                         │
├─────────────────────────┤
│   POLITE 徽章   [1/5]    │   ← card 1..N：英语表达（右上角进度）
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

## 工作流（一键端到端）

```
scripts/output/N.json
       │
       │ scripts/render.ts ──────────────┐
       │   ① desc   ② assets            │
       │   ③ audio  ④ video             │
       │   --clean 可选（清理中间产物）  │
       ▼                                 ▼
scripts/desc/N.draft.json         out/N-desc.mp4
public/audio/N/*.mp3              （唯一产物，或配合 --clean 仅留此）
public/images/N.jpg
```

**为什么用 render.ts**：把所有"生成产物 + 渲染"的步骤打包成一条命令，支持：
- **分阶段**：单独跑 desc/assets/audio/video 任意一步
- **强制重生成**： `--force` 跳过已存在检测
- **自动清理**： `--clean` 渲染后只留 mp4
- **进度可视化**：每阶段打印 ✅ / ❌

## 快速开始

```bash
cd scripts/video

# 1. 装依赖（pnpm 10+）
pnpm install && pnpm rebuild esbuild

# 2. 设置 API key（~/.zshrc 已配也行）
export MINIMAX_API_KEY=eyJhbGc...

# 3. 端到端跑一份 desc 1.json（一条命令搞定）
pnpm exec tsx scripts/render.ts scripts/output/1.json

# 4. 打开视频
open out/1-desc.mp4
```

## 详细命令

### render.ts · 端到端流水线

```bash
# 端到端：desc + assets + audio + video
pnpm exec tsx scripts/render.ts scripts/output/1.json

# 强制重新生成所有资产 + 渲染后清理（只留 mp4）
pnpm exec tsx scripts/render.ts scripts/output/1.json --force --clean

# 只跑某一阶段
pnpm exec tsx scripts/render.ts scripts/output/1.json --phase desc      # 只生成 desc JSON
pnpm exec tsx scripts/render.ts scripts/output/1.json --phase assets    # 只生成场景插图
pnpm exec tsx scripts/render.ts scripts/output/1.json --phase audio     # 只生成所有卡片音频
pnpm exec tsx scripts/render.ts scripts/output/1.json --phase video     # 只渲染视频（依赖已有 desc）

# 直接渲染指定 desc JSON（跳过前几个阶段）
pnpm exec tsx scripts/render.ts --phase video --desc scripts/desc/1.draft.json

# 清理时保留部分产物
pnpm exec tsx scripts/render.ts scripts/output/1.json --clean --keep-images   # 保留图片
pnpm exec tsx scripts/render.ts scripts/output/1.json --clean --keep-audio    # 保留音频
pnpm exec tsx scripts/render.ts scripts/output/1.json --clean --keep-desc     # 保留 desc JSON

# 自定义输出路径
pnpm exec tsx scripts/render.ts scripts/output/1.json -o out/custom.mp4
```

参数全览：`pnpm exec tsx scripts/render.ts --help`

### 预览（Remotion Studio）

```bash
cd scripts/video

# 预览默认 desc（1.draft.json）
pnpm studio

# 预览指定 desc 文件
pnpm studio --props scripts/desc/6.draft.json

# 预览指定 id 的 desc
pnpm studio --props scripts/desc/$(ls scripts/desc/ | fzf).draft.json
```

Remotion Studio 会启动浏览器预览，可实时看到视频效果。传入 `--props` 时会使用对应的 desc JSON 数据（场景、任务、英语表达等）。

### 独立脚本（保留兼容）

| 命令 | 作用 |
|------|------|
| `pnpm exec tsx scripts/generate-desc.ts [源JSON]` | 单独生成 desc JSON（默认 `../output/1.json`） |
| `pnpm exec tsx scripts/preview-card.ts [desc.json] [cardIdx]` | 生成 HTML 预览（默认 card 1） |
| `pnpm exec tsx scripts/generate-card-audio.ts [desc.json] [cardIdx\|all]` | TTS 单/全部卡片音频 + 回填 duration |
| `pnpm exec tsx scripts/generate-all.ts [desc.json] [--force]` | 一键生成所有音频 + 场景插画（不用 render.ts 的等价物） |
| `pnpm exec tsx scripts/update-durations.ts [desc.json]` | 用新公式（含段间停顿）重算 duration_sec |
| `pnpm exec remotion render src/index.ts EnSentenceVideo out/X.mp4 --props=<json>` | 直接调 Remotion CLI（render.ts 内部用） |
| `pnpm exec remotion studio src/index.ts` | Remotion Studio（实时预览） |
| `pnpm typecheck` | TypeScript 类型检查 |

## 文件结构

```
scripts/video/
├── src/
│   ├── index.ts                       # registerRoot
│   ├── Root.tsx                       # Compositions 注册
│   ├── Video.tsx                      # 主组合：Sequence 拼接 + 全局 Header/Footer
│   ├── HelloWorld.tsx                 # demo（Step 1）
│   ├── Card1Test.tsx                  # Step 8 回归测试
│   ├── theme.ts                       # 全局主题常量
│   ├── components/
│   │   ├── Header.tsx                 # 顶部条（"蛙蛙英语口语"）
│   │   ├── Footer.tsx                 # 底部条
│   │   └── cards/
│   │       ├── IntroCard.tsx          # 封面卡（图 + DAY N + SCENE/TASK）
│   │       ├── ExpressionCard.tsx     # 表达卡（badge/i/M 进度/zh/en/phonetic/note）
│   │       └── SummaryCard.tsx        # 总结卡（eyebrow + explanation）
│   └── api/
│       ├── minimax-tts.ts             # TTS 客户端
│       └── minimax-image.ts           # text_to_image 客户端
│
├── scripts/                           # CLI 脚本
│   ├── render.ts                      # 🆕 Step 10 端到端流水线（desc/assets/audio/video + --clean）
│   ├── generate-desc.ts               # 本地 desc 生成器
│   ├── preview-card.ts                # HTML 卡片预览
│   ├── generate-card-audio.ts         # TTS + duration 回填（单/全部）
│   ├── generate-all.ts                # 全量资产生成
│   └── update-durations.ts            # 用新公式重算 duration
│
├── public/
│   ├── audio/N/                       # TTS 音频（按 desc id 分目录）
│   └── images/N.jpg                   # 场景插画（1280×720 横版）
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

### Header / Footer / DAY N

`src/Video.tsx` 里的默认值：

```tsx
<Header text="蛙蛙英语口语" ... />                  // 默认 brand
<Footer text="@en-sentence-study" ... />             // 默认 handle

const dayNumber = parseInt(desc.id.match(/(\d+)$/)?.[1] ?? '0', 10) || 1;
```

`dayNumber` 从 desc JSON 的 `id` 字段末尾数字自动提取（`1.json` → `DAY 1`，`49.json` → `DAY 49`）。

### 段间停顿

`src/theme.ts`：

```ts
PAUSE_MS = 700;   // zh → en 之间的静音
```

要调整同时改 `generate-card-audio.ts` / `generate-all.ts` / `render.ts` 里的同名常量。

## 渲染参数

`<Composition>`（`src/Root.tsx`）：

```tsx
durationInFrames={defaultDesc.duration_frames}   // 由 desc JSON 提供（动态）
fps={defaultDesc.fps}                            // 默认 30
width={1080}
height={1920}
```

`calculateMetadata` 让 CLI `--props` 喂不同 desc 时 duration 自动适配。

## 已知限制 / 避坑

| 坑 | 解决方案 |
|----|---------|
| pnpm 不跑 esbuild postinstall | `pnpm rebuild esbuild` |
| MiniMax 签名 URL 时钟漂移 403 | `text_to_image` 默认 base64 模式 |
| `<Audio>` 默认从父 frame 0 播 → 多段重叠 | 每段 Audio 包 `<Sequence from={...}>` |
| 卡片 outer `background: c.bg` 挡死全局 bg | 卡片不要设 bg，让全局 bg 透过来 |
| overlay alpha > 0.5 把 bg 图挡死 | 保持 0.2-0.25 |
| duration_sec 必须含段间停顿 | 用 `update-durations.ts` 或新公式 |
| 横版图 `width: 'auto'` 会溢出裁剪 | 用 `width: 90% + height: 自适应`（IntroCard 已修） |
| 跨目录 CLI 的相对路径 cwd 错乱 | `resolve(PROJECT_ROOT, input)` 显式指定基准 |
| `--clean` 后再跑 video 会缺图片/音频 | 跑 `--force` 或重新跑 `--phase assets/audio` |

## 路线图

- [x] Step 1-9：环境 → 组件 → 端到端渲染（详情见 `docs/video-generator-progress.md`）
- [x] **Step 10**：`scripts/render.ts` CLI（desc/assets/audio/video 分阶段 + `--force` + `--clean`）
- [x] **Step 10.1**：UI 升级（蛙蛙英语口语 / DAY N / i/M 进度）
- [x] **Step 10.2**：图片裁剪问题最终修复（90% 宽 + 自适应高）
- [ ] **Step 11**：批量跑 49 个 desc → 49 个 mp4
- [ ] **Step 12**：LLM 优化文案（人话化 TTS / 增加 expressions）