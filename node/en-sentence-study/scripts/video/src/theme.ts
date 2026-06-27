// 视频全局主题常量 + 工具函数
// 共享给 Header / Footer / 三种 Card，避免重复硬编码

export const LAYOUT = {
  headerHeight: 110,
  footerHeight: 90,
  paddingX: 80,
  paddingY: 100,
} as const;

export const THEME_COLORS = {
  mint: {
    bg: '#F4FBF8',
    text: '#0E3B2E',
    subtext: '#5A8475',
    accent: '#19A974',
    border: '#CFE9DC',
    noteBg: 'rgba(207, 233, 220, 0.4)',
  },
  sunny: {
    bg: '#FFF8EB',
    text: '#3E2A14',
    subtext: '#9A7B53',
    accent: '#F58A1F',
    border: '#F6DDB5',
    noteBg: 'rgba(246, 221, 181, 0.4)',
  },
} as const;

export type Theme = keyof typeof THEME_COLORS;

// ── Expression style 配色（POLITE / NEUTRAL / CASUAL / BOLD）──
export const STYLE_COLORS = {
  polite:  { bg: '#A8E6CF', text: '#0E3B2E', label: 'POLITE · 礼貌' },
  neutral: { bg: '#CFE4F5', text: '#0E3B2E', label: 'NEUTRAL · 中性' },
  casual:  { bg: '#FFD6A8', text: '#3E2A14', label: 'CASUAL · 口语' },
  bold:    { bg: '#F5B7C5', text: '#3E2A14', label: 'BOLD · 直接' },
} as const;

export type StyleKind = keyof typeof STYLE_COLORS;

// ── TTS segment 类型（卡片通用）──
export type TtsSegment = {
  lang: 'zh' | 'en';
  text: string;
  audio_path: string;
  duration_ms: number;
};

// ── 全局字体 ──
export const FONT_FAMILY =
  'system-ui, -apple-system, "PingFang SC", "Helvetica Neue", "Microsoft YaHei", sans-serif';

// ── 工具：把绝对路径转成 staticFile() 用的相对路径 ──
// 例：/Users/.../scripts/video/public/audio/1/1-0-zh.mp3 → audio/1/1-0-zh.mp3
// 例：/Users/.../scripts/video/public/images/1.jpg       → images/1.jpg
export const toStaticFile = (absolutePath: string): string => {
  const idx = absolutePath.indexOf('/public/');
  return idx === -1 ? absolutePath : absolutePath.slice(idx + '/public/'.length);
};

// ── TTS 段间停顿（zh 播完后到 en 开始前的空白）──
// 共享给 ExpressionCard 排版 + update-durations.ts 重算 duration_sec
export const PAUSE_MS = 700;
export const PAUSE_FRAMES_AT_30FPS = Math.round((PAUSE_MS / 1000) * 30); // 21