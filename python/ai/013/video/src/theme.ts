// XOR 视频全局主题常量 + 工具函数
// 共享给 Header / Footer / 三种 Card,避免重复硬编码

// ── 横版 1920×1080 LAYOUT(Plan agent 建议调整,1080 高下原数值偏小) ──
export const LAYOUT = {
  headerHeight: 90,
  footerHeight: 70,
  paddingX: 140,
  paddingY: 60,
} as const;

// ── 主题色(白底深字,适合数学公式 + 暖橙高亮) ──────────────
export const THEME_COLORS = {
  paper: {
    bg: '#FFFFFF',
    text: '#1A1A1A',
    subtext: '#5C5C5C',
    accent: '#E07B00',       // 暖橙,公式高亮
    border: '#E5E5E5',
    formulaBg: '#F7F7F5',
    highlight: '#FFE7CC',
    codeBg: '#1E1E1E',
    codeText: '#E6E6E6',
    codeHighlight: '#3B3B3B',
    codeLineHighlight: '#4A4A4A',
  },
} as const;

export type Theme = keyof typeof THEME_COLORS;

// ── TTS segment 类型(全中文,统一) ─────────────────────────
export type TTSLanguage = 'zh';

export type TtsSegment = {
  lang: TTSLanguage;
  text: string;
  audio_path: string;
  duration_ms: number;
};

// ── FrameStep 类型(PNG 序列动画,frame index 触发) ─────────────
export type FrameStep = {
  step: number;
  start_frame: number;        // card 内相对 frame (从 0 开始)
  end_frame: number;
  target: string;             // 语义锚点,如 "highlight_yhat_minus_y"
  image: string;              // 该步骤的预渲染 PNG (相对路径)
};

// ── LatexStep 类型(KaTeX 实时渲染的序列动画) ─────────────
export type LatexStep = {
  start_frame: number;
  end_frame: number;
  latex: string;              // 该步骤的 LaTeX 字符串(已转义 `\` 为 `\\`)
  caption?: string;           // 可选,行内说明
  highlight_keys?: string[];  // 形如 "x",渲染时找 latex 内 {{x}}...{{/x}} 区间
};

// ── Card payload 类型 ────────────────────────────────────
export type CardPayload = {
  image?: string;             // 相对路径
  caption?: string;           // 底部字幕, ≤18 字,与 TTS 同步淡入
  steps?: FrameStep[];        // math_anim PNG 模式: 序列帧(image)
  latex_steps?: LatexStep[];  // math_anim KaTeX 模式: 序列帧(latex)
  headline?: string;          // 大字标题(intro/diagram/plot/formula/code)
  body?: string;              // 段落文字(intro 描述)
  code_text?: string;         // 代码 (等宽字体)
  code_highlight_lines?: number[];  // 高亮行号
};

// ── 顶层 card 类型 ────────────────────────────────────────
export type DescCard = {
  index: number;
  type: 'intro' | 'formula' | 'math_anim' | 'plot' | 'diagram' | 'code';
  duration_sec: number;
  tts_segments: TtsSegment[];
  payload?: CardPayload;
};

// ── Desc JSON ─────────────────────────────────────────────
export type DescJson = {
  id: string;
  fps: number;
  width: number;
  height: number;
  theme: Theme;
  meta: {
    title: string;
    subtitle: string;
  };
  cards: DescCard[];
  duration_sec: number;
  duration_frames: number;
};

// ── RichText 行内数学字符串占位符 `[[latex:..]]` 解析的 segment —— 给 RichText 组件用
export type RichTextSegment =
  | { kind: 'text'; value: string }
  | { kind: 'inline_math'; value: string };
export type RichTextInput = string | RichTextSegment[];

// ── 字体(PingFang SC 优先,Web 端 pingfang 不存在时回退系统字体) ──
export const FONT_FAMILY =
  '"PingFang SC", "Helvetica Neue", "Microsoft YaHei", system-ui, -apple-system, sans-serif';

export const MONO_FONT =
  '"JetBrains Mono", "SF Mono", "Menlo", "Consolas", monospace';

// ── 工具:把绝对路径转成 staticFile() 用的相对路径 ──
export const toStaticFile = (absolutePath: string): string => {
  if (!absolutePath) return absolutePath;
  // pipeline/output 的绝对路径也可能 (e.g. /xxx/public/audio/...)
  const idx = absolutePath.indexOf('/public/');
  return idx === -1 ? absolutePath : absolutePath.slice(idx + '/public/'.length);
};

// 工具:把 assets/xxx.png 转成 public/images/xxx.png(同步到 render 拷贝结果)
export const assetToStaticFile = (imageRef: string): string => {
  // imageRef 形如 "assets/diagrams/01-network.png"
  const filename = imageRef.split('/').pop() ?? '';
  return `images/${filename}`;
};
