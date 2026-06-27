# 英语口语视频生成器（Remotion + MiniMax 多模态）

## Context

**为什么做这个项目**：
用户已有 49 个 JSON 场景口语学习数据（`scripts/output/1.json` ~ `49.json`），现在要把它们自动转成手机端可看的英语口语讲解短视频——每个 JSON 生成一段 mp4，流程是：介绍场景/任务 → 每个英文表达一张卡片 → 总结。

**核心痛点**：
- 项目目前是纯 Python 数据生成工具，没有 Node.js 工程
- 49 条数据纯文本，需要视觉化包装才能让人愿意刷
- 用户希望"先生成描述 JSON 人工审核 → 再渲染视频"，避免 LLM 幻觉直接烧成视频浪费时间

**约束（来自用户）**：
- pnpm，不用 npm
- 中文显示 plan
- 亮色调（清新薄荷 / 暖阳橙黄两套主题）
- 9:16 竖屏（1080×1920）
- Header + Footer 每帧必有
- API key 从环境变量 `MINIMAX_API_KEY` 读取（已配在 ~/.zshrc）
- **两阶段工作流，无 all**：必须分开 `video:desc` 和 `video:render`，人工审核是必经环节
- **每场景生成一张 AI 插画**（不是纯文字，也不是每张卡都生成）
- **中英文全部走 TTS**（intro/expression/summary 都朗读）

**API 资源**：
- MiniMax TTS: https://platform.minimaxi.com/docs/api-reference/speech-t2a-http
- MiniMax Anthropic 兼容 LLM: https://platform.minimaxi.com/docs/api-reference/text-anthropic-api
- MiniMax text_to_image: HTTP API（与 LLM/TTS 同 base URL，参考 MCP `MiniMax0__text_to_image` 的响应格式）
- 提示：text_to_image 返回的 URL 含 URL 编码（`%2F`），需要 `decodeURIComponent` 还原

## 工作目录与环境

- 工作目录：`/Users/huhao/src/codesnip/node/en-sentence-study/`
- 系统：macOS 26.3 (arm64)
- Python 已配（用于现有 `scripts/generate_sentence.py`，不动）
- Node.js（系统已有，pnpm 由用户提供）
- 视频画幅：1080×1920 @ 30fps
- Remotion 自带 ffmpeg，无需系统装

## 项目结构（新增 `scripts/video/` 子项目，与 Python 工具链隔离）

```
/Users/huhao/src/codesnip/node/en-sentence-study/
├── docs/
├── scripts/
│   ├── generate_sentence.py        # 既有 Python，零修改
│   ├── llm_caller.py
│   ├── markdown_parser.py
│   ├── output/                     # 既有 N.json 数据
│   │
│   └── video/                      # ★ 新增 Remotion 子项目
│       ├── package.json
│       ├── tsconfig.json
│       ├── remotion.config.ts
│       ├── .env.example
│       │
│       ├── src/
│       │   ├── index.ts            # CLI 入口（desc / render 两个子命令，无 all）
│       │   ├── Root.tsx
│       │   ├── Video.tsx
│       │   │
│       │   ├── components/
│       │   │   ├── Header.tsx
│       │   │   ├── Footer.tsx
│       │   │   ├── IntroCard.tsx
│       │   │   ├── ExpressionCard.tsx
│       │   │   ├── SummaryCard.tsx
│       │   │   ├── SceneImage.tsx        # 渲染每场景一张的 AI 插画
│       │   │   └── BackgroundDecor.tsx
│       │   │
│       │   ├── config/
│       │   │   ├── video.config.ts       # 尺寸/帧率/默认时长/header/footer
│       │   │   ├── themes.ts             # mint / sunny 两套亮色主题
│       │   │   ├── tts.config.ts         # 中英双 voice_id
│       │   │   ├── llm.config.ts
│       │   │   ├── image.config.ts       # text_to_image 默认参数
│       │   │   └── prompts/
│       │   │       ├── desc.template.ts  # 描述 JSON 生成 prompt
│       │   │       └── image.template.ts # 插画 prompt
│       │   │
│       │   ├── stages/
│       │   │   ├── stageA-desc.ts        # 源 JSON → 描述 JSON（draft）
│       │   │   └── stageB-render.ts      # 描述 JSON → mp4
│       │   │
│       │   ├── api/
│       │   │   ├── minimax-tts.ts
│       │   │   ├── minimax-llm.ts
│       │   │   ├── minimax-image.ts      # text_to_image HTTP 调用
│       │   │   └── audio-decode.ts
│       │   │
│       │   ├── schema/
│       │   │   ├── source.ts             # zod: 源 N.json
│       │   │   └── description.ts        # zod: 描述 JSON（含 image_url）
│       │   │
│       │   └── utils/
│       │       ├── logger.ts
│       │       ├── paths.ts
│       │       └── retry.ts
│       │
│       ├── public/
│       │   ├── audio/silent.mp3          # TTS 失败占位（0.5s 静音）
│       │   └── images/.gitkeep           # AI 插画缓存（首次下载后本地复用）
│       │
│       └── scripts/                      # 运行时产物（gitignore）
│           ├── desc/<id>.draft.json
│           ├── desc/<id>.approved.json
│           ├── audio/<id>/<idx>.mp3
│           ├── images/<id>.jpg           # 下载到本地的插画
│           └── out/<id>.mp4
│
└── .gitignore (新增)
```

**为什么用子目录而不是根目录**：项目原本是 Python 工具链，根目录无 package.json；Remotion 子项目与 Python 工具链解耦，避免相互污染。

## 关键技术决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Node 工程位置 | `scripts/video/` 子目录 | 与 Python 工具链物理隔离 |
| 工作流拆分 | 仅 `desc` + `render` 两个子命令 | 用户明确要求"人工审核后再生成视频" |
| 视频框架 | Remotion 4.x | React 生态、声明式、@remotion/renderer 可无头渲染 |
| 渲染 API | `@remotion/renderer` Node API（`renderMedia` + `bundle`） | CLI 友好，无需 spawn 子进程 |
| 字体 | `@remotion/google-fonts` 加载 Noto Sans SC + Inter | 中英文覆盖，无水合闪烁 |
| HTTP 客户端 | Node 22 内置 `fetch` | 零依赖 |
| 类型校验 | `zod` | 源 JSON / 描述 JSON 双向校验 |
| LLM 客户端 | `@anthropic-ai/sdk` + 自定义 `baseURL` | 复用 Anthropic SDK 调用 MiniMax 兼容层 |
| JSON 强约束 | tool_use（Anthropic 协议原生） | 比 `response_format` 更可靠 |
| TTS 编码处理 | hex → Buffer（自动嗅探 base64） | 文档说 hex，实际有时返回 base64 |
| AI 插画 | 每场景一张，通过 text_to_image HTTP API | 用户指定；URL 需 `decodeURIComponent` 还原 %2F |
| 插画位置 | 全部卡片背景 + IntroCard 前景焦点 | 视觉一致，避免单调 |
| 主题 | mint / sunny 两套亮色预设 | 用户偏好；可在 config 切换 |
| 中英 TTS | 配两个 voice_id（zh / en） | 用户要求中英文全部朗读 |
| 卡片时长 | 默认值 + 每卡可覆盖（draft JSON 编辑） | 灵活但不繁琐 |

## 依赖清单（pnpm）

**dependencies**：
```
pnpm add remotion @remotion/cli @remotion/renderer \
         react react-dom \
         @remotion/google-fonts \
         @anthropic-ai/sdk \
         zod commander chalk ora
```

**devDependencies**：
```
pnpm add -D typescript @types/react @types/react-dom @types/node \
            tsx dotenv vitest
```

**不安装的依赖**：
- ❌ axios / undici（Node 22 fetch 足够）
- ❌ ffmpeg 系统包（Remotion 内置 bundled ffmpeg）
- ❌ framer-motion（Remotion 自带 `spring`/`interpolate`）

## 配置系统（src/config/）

### video.config.ts

```ts
export const VIDEO_CONFIG = {
  width: 1080,
  height: 1920,
  fps: 30,

  defaultDurations: {
    intro: 4,        // 场景任务介绍
    expression: 6,   // 每张英文表达卡
    summary: 6,      // 总结
  },

  header: {
    text: '英语口语 · 每日一句',
    bgColor: 'transparent',
    textColor: 'theme.text',
    fontSize: 36,
    height: 110,
    showOn: ['intro', 'expression', 'summary'],
  },

  footer: {
    prefix: '@en-sentence-study',
    bgColor: 'transparent',
    textColor: 'theme.textMuted',
    fontSize: 28,
    height: 90,
    showOn: ['intro', 'expression', 'summary'],
  },

  defaultTheme: 'mint' as const,  // 'mint' | 'sunny'
};
```

### themes.ts（两套亮色）

```ts
export type ThemeName = 'mint' | 'sunny';

export const THEMES: Record<ThemeName, Theme> = {
  mint: {
    name: 'mint',
    bg: '#F4FBF8',
    bgGradient: ['#E6F7F0', '#F4FBF8'],
    text: '#0E3B2E',
    textMuted: '#5A8475',
    accent: '#19A974',
    cardBg: '#FFFFFF',
    cardBorder: '#CFE9DC',
    highlight: '#FFE8A3',
  },
  sunny: {
    name: 'sunny',
    bg: '#FFFBF2',
    bgGradient: ['#FFF1D6', '#FFFBF2'],
    text: '#3E2A14',
    textMuted: '#9A7B53',
    accent: '#F58A1F',
    cardBg: '#FFFFFF',
    cardBorder: '#F6DDB5',
    highlight: '#FFE08A',
  },
};

export const STYLE_ACCENT: Record<string, { light: string; dark: string }> = {
  polite:  { light: '#A8E6CF', dark: '#19A974' },
  neutral: { light: '#CFE4F5', dark: '#3B82C4' },
  casual:  { light: '#FFD6A8', dark: '#F58A1F' },
  bold:    { light: '#F5B7C5', dark: '#D6336C' },
};
```

### tts.config.ts（中英双 voice_id）

```ts
export const TTS_CONFIG = {
  endpoint: 'https://api.minimaxi.com/v1/t2a_v2',
  apiKeyEnv: 'MINIMAX_API_KEY',
  groupIdEnv: 'MINIMAX_GROUP_ID',  // 可选
  model: 'speech-02-hd',
  speed: 1.0,
  vol: 1.0,
  pitch: 0,
  audioFormat: 'mp3',
  sampleRate: 32000,
  bitrate: 128000,
  maxRetries: 3,

  // ★ 用户要求中英文都朗读，所以配两个 voice_id
  voices: {
    zh: 'female-shaonv',           // 中文女声
    en: 'English_PassionateWarrior', // 英文男声（激情向，适合口语教学）
    // 也可改: 'English_Graceful_Lady' / 'male-qn-jingying'
  },
};
```

### llm.config.ts

```ts
export const LLM_CONFIG = {
  baseUrl: 'https://api.minimaxi.com/anthropic',
  apiKeyEnv: 'MINIMAX_API_KEY',
  model: 'MiniMax-Text-01',       // 与现有 Python 工具链保持一致
  maxTokens: 4096,
  temperature: 0.4,
  maxRetries: 3,
};
```

### image.config.ts（text_to_image）

```ts
export const IMAGE_CONFIG = {
  endpoint: 'https://api.minimaxi.com/v1/text_to_image',  // 待与 MCP 工具响应格式核对
  apiKeyEnv: 'MINIMAX_API_KEY',
  model: 'image-01',
  aspectRatio: '9:16',            // 匹配视频画幅
  width: 1024,
  height: 1824,                   // 略小于 1080×1920，留边距
  maxRetries: 3,
  cacheDir: 'public/images',      // 缓存已生成的插画
  // 透明白底，方便作为卡片底层
  background: 'white',
};
```

### prompts/desc.template.ts

```ts
export const DESC_SYSTEM_PROMPT = `你是一名英语教学视频脚本编辑。给定一段"场景 + 中文句 + 多个英文表达"，
为视频的每一帧生成画面文案、TTS 朗读文本、动画提示。要求：
1. 中文场景介绍 ≤ 40 字，作为视频开场朗读稿。
2. 每条英文表达对应一张卡片：tts_text = sentence 原句；caption = ≤ 20 字中文画面提示。
3. 总结卡片 tts_text 用 explanation 改写成 ≤ 80 字中文口语稿。
4. 为整个场景生成一个 image_prompt（≤ 60 词英文），用于 text_to_image 生成场景插画。
   风格要求：扁平插画、明亮色调、白底、9:16 竖构图、无文字、无真人。
5. 仅返回严格 JSON（通过 tool_use），不允许任何额外文本。`;

export const DESC_USER_TEMPLATE = `场景：{{scene_zh}} ({{scene_en}})
任务：{{task_zh}} ({{task_en}})
中文原句：{{sentence_zh}}
上下文：{{context}}

英文表达：
{{translations_block}}

讲解：{{explanation}}

请按 system 要求返回 JSON。`;
```

### prompts/image.template.ts

```ts
export const IMAGE_USER_TEMPLATE = `风格：flat illustration, vector art, minimalist,
bright color palette, light background, no text, no real people, no watermark,
portrait orientation 9:16.

场景：{{image_prompt}}`;
```

## 视频描述 JSON Schema（src/schema/description.ts）

```ts
import { z } from 'zod';

export const CardVisual = z.object({
  layout: z.enum(['title-big', 'split', 'caption-bottom']),
  bg_color: z.string(),
  accent_text: z.string().optional(),
  caption: z.string().max(40).optional(),
});

export const Card = z.object({
  index: z.number().int(),
  type: z.enum(['intro', 'expression', 'summary']),
  duration_sec: z.number().positive(),
  header_text: z.string().default(''),
  footer_text: z.string().default(''),

  // TTS（每个 card 都用，按 language 自动选 voice_id）
  tts_text: z.string(),
  tts_language: z.enum(['zh', 'en']),
  tts_voice_id: z.string().optional(),  // 不填则用 tts.config 默认
  tts_audio_path: z.string().nullable(), // stageA 留空，stageB 填入

  // 视觉
  visual: CardVisual,

  // 仅 expression 卡片有
  style: z.enum(['polite', 'neutral', 'casual', 'bold']).optional(),
  sentence_en: z.string().optional(),
  phonetic: z.string().optional(),
  literal_translation: z.string().optional(),
  note: z.string().optional(),
});

export const HeaderCfg = z.object({
  text: z.string(),
  bgColor: z.string(),
  textColor: z.string(),
  fontSize: z.number(),
  height: z.number(),
});

export const Meta = z.object({
  scene_en: z.string(),
  scene_zh: z.string(),
  task_en: z.string(),
  task_zh: z.string(),
  context: z.string(),
  explanation: z.string(),
  header: HeaderCfg,
  footer: FooterCfg,
});

export const DescriptionJSON = z.object({
  id: z.string(),                       // "1"
  source_json: z.string(),
  fps: z.number().int(),
  duration_frames: z.number().int(),
  theme: z.enum(['mint', 'sunny']),
  meta: Meta,
  // ★ 每场景一张 AI 插画
  scene_image: z.object({
    prompt: z.string(),                 // 喂给 text_to_image 的英文 prompt
    url: z.string().nullable(),         // stageA 调用后填入
    local_path: z.string().nullable(),  // stageB 下载到本地后填入
  }),
  cards: z.array(Card).min(2),
});
```

**示例（id=1 简版）**：
```json
{
  "id": "1",
  "source_json": "scripts/output/1.json",
  "fps": 30,
  "duration_frames": 750,
  "theme": "mint",
  "meta": { "scene_en": "Shopping", "scene_zh": "购物", ... },
  "scene_image": {
    "prompt": "flat illustration of a shopping mall checkout counter with gift boxes, bright colors",
    "url": "https://...",
    "local_path": null
  },
  "cards": [
    { "index": 0, "type": "intro", "duration_sec": 4,
      "tts_text": "今天我们学习购物场景：礼品包装。", "tts_language": "zh",
      "tts_audio_path": null,
      "visual": { "layout": "title-big", "bg_color": "#F4FBF8",
                  "accent_text": "购物 · 礼品包装" } },
    { "index": 1, "type": "expression", "duration_sec": 6,
      "tts_text": "Could you gift wrap this, please?", "tts_language": "en",
      "visual": { "layout": "split", "caption": "最常用且礼貌" },
      "style": "polite",
      "sentence_en": "Could you gift wrap this, please?",
      "phonetic": "/kuːd juː ɡɪft ræp ðɪs pliːz/",
      "literal_translation": "您能把这个礼品包装一下吗？",
      "note": "最常用且礼貌的说法，gift wrap 是固定搭配" },
    ...
    { "index": 6, "type": "summary", "duration_sec": 6,
      "tts_text": "在购物结账时请求礼品包装……", "tts_language": "zh",
      "visual": { "layout": "caption-bottom" } }
  ]
}
```

## Remotion 组件分层

### Root.tsx

```ts
import { Composition, CalculateMetadataFunction } from 'remotion';
import { Video } from './Video';
import { VIDEO_CONFIG } from './config/video.config';

export const RemotionRoot: React.FC = () => (
  <Composition
    id="en-sentence-video"
    component={Video}
    durationInFrames={VIDEO_CONFIG.defaultDurations.intro
      + 5 * VIDEO_CONFIG.defaultDurations.expression
      + VIDEO_CONFIG.defaultDurations.summary}
    fps={VIDEO_CONFIG.fps}
    width={VIDEO_CONFIG.width}
    height={VIDEO_CONFIG.height}
    calculateMetadata={({ props }) => ({
      durationInFrames: (props as any).desc.duration_frames,
      props,
    })}
  />
);
```

### Video.tsx（Sequence 拼接 + 全局 Header/Footer）

```ts
import { Sequence, AbsoluteFill, Audio } from 'remotion';

export const Video: React.FC<{ desc: DescriptionJSON }> = ({ desc }) => {
  const { fps } = useVideoConfig();
  let cursor = 0;
  return (
    <AbsoluteFill style={{ background: 'white' }}>
      {desc.cards.map((card) => {
        const dur = Math.round(card.duration_sec * fps);
        const start = cursor;
        cursor += dur;
        return (
          <Sequence key={card.index} from={start} durationInFrames={dur}>
            {/* 场景插画作底层背景（透明度 0.18） */}
            <SceneImage image={desc.scene_image} dim={0.18} />
            {/* 主题渐变背景 */}
            <BackgroundDecor card={card} theme={desc.theme} />
            {/* 全局 Header/Footer */}
            <Header cfg={desc.meta.header} theme={desc.theme} />
            {/* 卡片内容 */}
            {card.type === 'intro'      && <IntroCard card={card} theme={desc.theme} />}
            {card.type === 'expression' && <ExpressionCard card={card} theme={desc.theme} />}
            {card.type === 'summary'    && <SummaryCard card={card} theme={desc.theme} />}
            <Footer cfg={desc.meta.footer} theme={desc.theme} />
            {/* TTS 音频 */}
            {card.tts_audio_path && <Audio src={card.tts_audio_path} />}
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
```

### 各组件 props + 动画

| 组件 | props | 动画 |
|------|-------|------|
| `Header` | `{ cfg, theme }` | 顶部固定条，spring 从 y=-110 → y=0（首 12 帧） |
| `Footer` | `{ cfg, theme }` | 底部固定条，前 12 帧淡入 + 上移 20px |
| `BackgroundDecor` | `{ card, theme }` | 主题渐变，0~20 帧淡入；卡片切换时颜色 crossfade |
| `SceneImage` | `{ image, dim }` | 静态模糊背景，`dim` 控制透明度 |
| `IntroCard` | `{ card, theme }` | 标题 spring 缩放 0.8→1；副标题延迟 12 帧淡入 |
| `ExpressionCard` | `{ card, theme }` | 英文打字机（`clipPath` 按字符展开）；phonetic+literal 12 帧后淡入；左侧色条按 `style` 配色 |
| `SummaryCard` | `{ card, theme }` | 中文段落逐句淡入（每 12 帧一句） |

### 字体加载

```ts
import { loadFont as loadCn } from '@remotion/google-fonts/NotoSansSC';
import { loadFont as loadEn } from '@remotion/google-fonts/Inter';
const { fontFamily: cn } = loadCn();
const { fontFamily: en } = loadEn();
// 中文用 cn、英文用 en
```

## API 客户端

### minimax-tts.ts

```ts
import { TTS_CONFIG } from '../config/tts.config';

export async function callTTS(opts: {
  text: string;
  language: 'zh' | 'en';
  voice_id?: string;
}): Promise<string> {  // 返回 hex
  const apiKey = requiredEnv(TTS_CONFIG.apiKeyEnv);
  const voiceId = opts.voice_id ?? TTS_CONFIG.voices[opts.language];
  const resp = await fetch(TTS_CONFIG.endpoint, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: TTS_CONFIG.model,
      text: opts.text,
      stream: false,
      voice_setting: {
        voice_id: voiceId,
        speed: TTS_CONFIG.speed, vol: TTS_CONFIG.vol, pitch: TTS_CONFIG.pitch,
      },
      audio_setting: {
        sample_rate: TTS_CONFIG.sampleRate,
        bitrate: TTS_CONFIG.bitrate,
        format: TTS_CONFIG.audioFormat,
      },
    }),
  });
  if (!resp.ok) throw new Error(`TTS HTTP ${resp.status}: ${await resp.text()}`);
  const json = await resp.json();
  if (json.base_resp?.status_code !== 0) {
    throw new Error(`TTS error: ${json.base_resp?.status_msg}`);
  }
  return json.data.audio as string;  // hex
}
```

### minimax-llm.ts（tool_use 强制 JSON）

```ts
import Anthropic from '@anthropic-ai/sdk';
import { LLM_CONFIG } from '../config/llm.config';

const DESC_TOOL_SCHEMA = {
  name: 'emit_video_desc',
  description: 'Emit a structured English-sentence video description.',
  input_schema: {
    type: 'object',
    required: ['intro', 'expressions', 'summary', 'image_prompt'],
    properties: {
      image_prompt: { type: 'string', maxLength: 200 },
      intro: {
        type: 'object',
        required: ['tts_text', 'caption', 'duration_sec'],
        properties: {
          tts_text: { type: 'string', maxLength: 200 },
          caption: { type: 'string', maxLength: 80 },
          duration_sec: { type: 'number' },
        },
      },
      expressions: {
        type: 'array',
        items: {
          type: 'object',
          required: ['tts_text', 'caption', 'duration_sec', 'style'],
          properties: {
            tts_text: { type: 'string' },
            caption: { type: 'string', maxLength: 60 },
            duration_sec: { type: 'number' },
            style: { enum: ['polite', 'neutral', 'casual', 'bold'] },
          },
        },
      },
      summary: {
        type: 'object',
        required: ['tts_text', 'duration_sec'],
        properties: {
          tts_text: { type: 'string', maxLength: 400 },
          duration_sec: { type: 'number' },
        },
      },
    },
  },
};

export async function callLLM(opts: { system: string; user: string }): Promise<any> {
  const client = new Anthropic({
    apiKey: process.env[LLM_CONFIG.apiKeyEnv]!,
    baseURL: LLM_CONFIG.baseUrl,
  });
  const msg = await client.messages.create({
    model: LLM_CONFIG.model,
    max_tokens: LLM_CONFIG.maxTokens,
    system: opts.system,
    tools: [DESC_TOOL_SCHEMA],
    tool_choice: { type: 'tool', name: 'emit_video_desc' },
    messages: [{ role: 'user', content: opts.user }],
  });
  const toolBlock = msg.content.find((b: any) => b.type === 'tool_use');
  if (!toolBlock) throw new Error('LLM 未返回 tool_use block');
  return toolBlock.input;
}
```

### minimax-image.ts（text_to_image + URL 解码）

```ts
import { IMAGE_CONFIG } from '../config/image.config';

export async function generateImage(opts: {
  prompt: string;
}): Promise<string> {  // 返回解码后的 URL
  const apiKey = requiredEnv(IMAGE_CONFIG.apiKeyEnv);
  const userPrompt = IMAGE_USER_TEMPLATE.replace('{{image_prompt}}', opts.prompt);
  const resp = await fetch(IMAGE_CONFIG.endpoint, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: IMAGE_CONFIG.model,
      prompt: userPrompt,
      aspect_ratio: IMAGE_CONFIG.aspectRatio,
      width: IMAGE_CONFIG.width,
      height: IMAGE_CONFIG.height,
    }),
  });
  if (!resp.ok) throw new Error(`Image HTTP ${resp.status}: ${await resp.text()}`);
  const json = await resp.json();
  // 假设响应形如 { data: { image_url: "https%3A%2F%2F..." } } 或 MCP 工具对应格式
  const rawUrl = json.data?.image_url ?? json.image_url ?? json.data?.[0]?.url;
  if (!rawUrl) throw new Error('未找到 image_url');
  // ★ CLAUDE.md 强调：URL 解码，%2F 变回 /
  return decodeURIComponent(rawUrl);
}
```

## CLI 入口（src/index.ts，**仅两个子命令，无 all**）

```ts
#!/usr/bin/env -S npx tsx
import 'dotenv/config';
import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import { stageA } from './stages/stageA-desc';
import { stageB } from './stages/stageB-render';

const program = new Command();
program
  .name('video')
  .description('英语口语视频生成 CLI（分两步：人工审核 → 渲染）')
  .version('0.1.0');

program
  .command('desc <sourceJson>')
  .description('阶段 A：源 JSON → 视频描述 JSON (draft) → 人工审核')
  .option('-t, --theme <name>', '主题: mint | sunny', 'mint')
  .action(async (src, opts) => {
    const spin = ora('生成视频描述 JSON（含 AI 插画 prompt 调用）...').start();
    try {
      const p = await stageA(src, { theme: opts.theme });
      spin.succeed(chalk.green(`描述已写入：${p}`));
      console.log(chalk.yellow('👉 请打开 draft 文件检查/编辑后，将其重命名为 .approved.json，再运行 render 子命令'));
    } catch (e) {
      spin.fail(chalk.red(String(e)));
      process.exit(2);
    }
  });

program
  .command('render <descJson>')
  .description('阶段 B：审核后的描述 JSON → mp4')
  .option('--skip-tts', '跳过 TTS（音频已存在时用）')
  .option('--skip-image', '跳过插画下载（已下载时用）')
  .action(async (desc, opts) => {
    const spin = ora('TTS + 渲染中...').start();
    try {
      const p = await stageB(desc, {
        skipTTS: opts.skipTTS,
        skipImage: opts.skipImage,
      });
      spin.succeed(chalk.green(`视频已生成：${p}`));
    } catch (e) {
      spin.fail(chalk.red(String(e)));
      process.exit(3);
    }
  });

// ★ 注意：没有 all 子命令——人工审核是必经环节

program.parseAsync().catch((e) => { console.error(e); process.exit(1); });
```

**package.json scripts**：
```json
{
  "scripts": {
    "studio": "remotion studio src/index.ts",
    "video:desc": "tsx src/index.ts desc",
    "video:render": "tsx src/index.ts render",
    "typecheck": "tsc --noEmit"
  }
}
```

## 阶段 A 逻辑（src/stages/stageA-desc.ts）

```
源 JSON ─► zod 校验 ─► 构造 user prompt ─► callLLM (tool_use)
                                          │
                              ┌─ 解析成功 ─┴─ 解析失败
                              ▼                ▼
                    拼接 description     fallback 模板
                    JSON（含 image_prompt）
                              │
                              ▼
                    ★ callImage(image_prompt)
                    → decodeURIComponent(URL)
                    → 填入 desc.scene_image.url
                              │
                              ▼
                    zod 二次校验 ─► 写入 desc/<id>.draft.json
```

**失败降级**：
- HTTP 错误：指数退避重试 3 次（1.5s/3s/6s）
- tool_use 解析失败：尝试从 text 块 ```json``` 抽取
- 仍失败：用本地 `fallbackDescription()` 生成最小可用描述
- 整阶段失败：CLI 退出码 2

## 阶段 B 逻辑（src/stages/stageB-render.ts）

```
approved desc JSON
  │
  ├─► 下载场景插画（如果本地缓存缺失）
  │     fetch(url) → 写 public/images/<id>.jpg → 填 local_path
  │
  ├─► 对每个 card 调用 TTS（按 tts_language 选 voice_id）
  │     ├─ 成功：hex → .mp3 → 填 tts_audio_path
  │     └─ 失败：用 silent.mp3 占位 + 仅动画
  │
  ├─► 校验所有音频文件存在
  │
  └─► @remotion/renderer.renderMedia
        - bundle({ entryPoint: 'src/Root.tsx' })
        - inputProps: { desc }
        - codec: 'h264'
        - concurrency: 1（1080×1920 内存敏感）
        → 输出 scripts/out/<id>.mp4
```

**失败降级**：
- TTS 单条失败：该卡用 silent 占位，整体继续
- 渲染失败：自动重试 1 次
- 渲染成功但文件 < 10KB：删除 + 退出码 3

## 关键文件清单

| 路径 | 职责 |
|------|------|
| `scripts/video/package.json` | pnpm 工程 |
| `scripts/video/tsconfig.json` | TS 配置（target ES2022、jsx react-jsx） |
| `scripts/video/remotion.config.ts` | Remotion CLI 配置 |
| `scripts/video/.env.example` | `MINIMAX_API_KEY=xxx` |
| `scripts/video/src/index.ts` | CLI 入口（仅 desc/render） |
| `scripts/video/src/Root.tsx` | Composition 注册 |
| `scripts/video/src/Video.tsx` | Sequence 拼接 |
| `scripts/video/src/components/Header.tsx` | 顶部条 |
| `scripts/video/src/components/Footer.tsx` | 底部条 |
| `scripts/video/src/components/IntroCard.tsx` | 场景任务介绍 |
| `scripts/video/src/components/ExpressionCard.tsx` | 单条表达 |
| `scripts/video/src/components/SummaryCard.tsx` | 总结 |
| `scripts/video/src/components/SceneImage.tsx` | AI 插画渲染层 |
| `scripts/video/src/components/BackgroundDecor.tsx` | 渐变背景 |
| `scripts/video/src/config/video.config.ts` | 视频尺寸/帧率/默认值 |
| `scripts/video/src/config/themes.ts` | mint / sunny |
| `scripts/video/src/config/tts.config.ts` | 中英双 voice_id |
| `scripts/video/src/config/llm.config.ts` | LLM 默认参数 |
| `scripts/video/src/config/image.config.ts` | text_to_image 默认参数 |
| `scripts/video/src/config/prompts/desc.template.ts` | 描述 JSON prompt |
| `scripts/video/src/config/prompts/image.template.ts` | 插画 prompt |
| `scripts/video/src/stages/stageA-desc.ts` | 阶段 A |
| `scripts/video/src/stages/stageB-render.ts` | 阶段 B |
| `scripts/video/src/api/minimax-tts.ts` | TTS |
| `scripts/video/src/api/minimax-llm.ts` | LLM (tool_use) |
| `scripts/video/src/api/minimax-image.ts` | text_to_image |
| `scripts/video/src/api/audio-decode.ts` | hex/base64 嗅探 |
| `scripts/video/src/schema/source.ts` | 源 JSON zod |
| `scripts/video/src/schema/description.ts` | 描述 JSON zod |
| `scripts/video/src/utils/paths.ts` | 路径常量 |
| `scripts/video/src/utils/retry.ts` | 指数退避 |
| `scripts/video/src/utils/logger.ts` | chalk/ora 包装 |
| `scripts/video/public/audio/silent.mp3` | TTS 失败占位 |
| `.gitignore`（根目录新增） | 忽略 node_modules、运行时产物 |

## 验证 / 测试方案

### 端到端冒烟（最快）

```bash
cd /Users/huhao/src/codesnip/node/en-sentence-study/scripts/video
pnpm install
cp .env.example .env   # 填入 MINIMAX_API_KEY

# 阶段 A
pnpm video:desc ../output/1.json
# 产物：scripts/desc/1.draft.json
# 用 jq '.cards | length' scripts/desc/1.draft.json → 至少 2（intro+summary），通常 7（1+N+1）
# 用 jq '.scene_image.url' scripts/desc/1.draft.json → 应为解码后的 URL

# 人工审核 → 重命名为 1.approved.json
mv scripts/desc/1.draft.json scripts/desc/1.approved.json
# 也可手动编辑后再改名

# 阶段 B
pnpm video:render scripts/desc/1.approved.json
# 产物：scripts/out/1.mp4
open scripts/out/1.mp4
```

### Remotion Studio 视觉验收

```bash
pnpm studio
# 浏览器打开 http://localhost:3000
# 用 Chrome DevTools MCP 验收：
#   - 1080×1920 9:16 正确
#   - Header / Footer 每帧可见
#   - 主题色对比度 ≥ 4.5:1
#   - 打字机动画连贯
#   - 卡片切换无白闪
#   - TTS 与字幕同步
```

### 单元测试（vitest）

| 范围 | 验证目标 |
|------|----------|
| `schema/source.ts` | 49 个 JSON 全部通过；坏 JSON 报错可读 |
| `schema/description.ts` | 阶段 A 产物必通过 |
| `api/audio-decode.ts` | hex / base64 / 异常输入 |
| `api/minimax-image.ts` | URL 解码 %2F → /（核心！CLAUDE.md 强调） |
| `utils/retry.ts` | 成功路径 + 全失败路径 |

不写 React 组件单测（动画由时间驱动，单元测试收益低），改用 Studio 视觉验收。

## 实施顺序

1. **Day 1**：`pnpm init` + package.json + tsconfig + 依赖安装 + `.env.example`
2. **Day 1**：`src/config/*` 全套配置 + `src/schema/*` zod schema
3. **Day 2**：`src/api/minimax-tts.ts` / `minimax-llm.ts` / `minimax-image.ts`（先 curl 验证 API 通）
4. **Day 2**：`src/stages/stageA-desc.ts` + `src/index.ts` 的 `desc` 子命令
5. **Day 3**：Remotion 组件（Header/Footer/Intro/Expression/Summary/SceneImage/BackgroundDecor）+ Video.tsx + Root.tsx
6. **Day 3**：`pnpm studio` 视觉验收
7. **Day 4**：`src/stages/stageB-render.ts` + `render` 子命令
8. **Day 4**：用 1.json 跑通端到端，输出 1.mp4
9. **Day 5**：单元测试 + 抽 3~5 个 JSON 验证批量可行性

## 风险与备选

| 风险 | 缓解 |
|------|------|
| MiniMax TTS 返回 base64 而非 hex | `audio-decode.ts` 自动嗅探 |
| Anthropic 兼容层不支持 tool_use | 降级：去掉 tools，用 ```json``` 围栏 + 正则抽取 |
| `@remotion/renderer` 在 arm64 缺 ffmpeg | Remotion 4.x 自带二进制；如报错提示 `brew install ffmpeg` |
| text_to_image URL 未编码 | 仍 `decodeURIComponent`（幂等） |
| 49 文件批量渲染内存爆 | `concurrency=1`；分批队列 |
| 中文 TTS 音色不够自然 | 配置 `voices.zh` 预留切换空间（`female-shaonv` → `male-qn-qingse` 等） |
| LLM 幻觉写出超长 tts_text | prompt 中明确 maxLength；description JSON zod 校验兜底 |

## 关键文件路径（实施时优先关注）

- `scripts/video/src/stages/stageA-desc.ts` — 阶段 A 核心：LLM → 描述 JSON + 调 text_to_image
- `scripts/video/src/stages/stageB-render.ts` — 阶段 B 核心：TTS + 下载插画 + Remotion 渲染
- `scripts/video/src/schema/description.ts` — 两阶段间唯一契约（含 scene_image 字段）
- `scripts/video/src/api/minimax-image.ts` — text_to_image + URL 解码（CLAUDE.md 强调）
- `scripts/video/src/api/minimax-tts.ts` — 中英双 voice_id TTS
- `scripts/video/src/config/prompts/desc.template.ts` — 描述 JSON 生成 prompt（所有可配置文案的源头）