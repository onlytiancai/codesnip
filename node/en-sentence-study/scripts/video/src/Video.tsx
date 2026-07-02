import { AbsoluteFill, Sequence } from 'remotion';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { IntroCard } from './components/cards/IntroCard';
import { ExpressionCard } from './components/cards/ExpressionCard';
import { SummaryCard } from './components/cards/SummaryCard';
import { THEME_COLORS, Theme } from './theme';

// ── desc JSON 类型（与 generate-desc.ts 输出对齐）──
export type TtsSegment = {
  lang: 'zh' | 'en';
  text: string;
  audio_path: string;
  duration_ms: number;
};

export type DescCard = {
  index: number;
  type: 'intro' | 'expression' | 'summary';
  duration_sec: number;
  tts_segments: TtsSegment[];
  style?: 'polite' | 'neutral' | 'casual' | 'bold';
  sentence_en?: string;
  phonetic?: string;
  literal_translation?: string;
  note?: string;
};

export type DescJson = {
  id: string;
  fps: number;
  theme: Theme;
  meta: {
    scene_en: string;
    scene_zh: string;
    task_en: string;
    task_zh: string;
    context: string;
    sentence_zh: string;
    explanation: string;
  };
  cards: DescCard[];
  scene_image: {
    prompt: string;
    url: string | null;
    local_path: string;
  };
  duration_sec: number;
  duration_frames: number;
};

export type VideoProps = {
  desc: DescJson;
  headerText?: string;
  footerText?: string;
};

// 兼容两种 props 格式：直接传 JSON 对象 或 包装在 { desc } 中
function resolveDesc(props: VideoProps): DescJson {
  // 已经是完整的 DescJson（直接传 JSON 对象时）
  if ('cards' in props && 'meta' in props) {
    return props as unknown as DescJson;
  }
  // 包装在 desc 属性中
  return (props as { desc: DescJson }).desc;
}

/**
 * 端到端视频主组合：
 *   - 底色（薄荷/暖橙）
 *   - 每张 card 用 <Sequence> 拼接到正确时间区间
 *   - 场景插画放进 IntroCard 内部作为主题图（不再做全局背景）
 */
export const Video: React.FC<VideoProps> = (props) => {
  const desc = resolveDesc(props);
  const { headerText, footerText } = props;
  // 从 desc.id 提取序号作为 Day N（id 可能是 "1" / "2" / "draft-1" 等，取最后一段数字）
  const dayNumber = parseInt(desc.id.match(/(\d+)$/)?.[1] ?? '0', 10) || 1;

  // 计算 expression cards 总数（用于进度 i/M）
  const expressionCards = desc.cards.filter((c) => c.type === 'expression');
  const expressionTotal = expressionCards.length;

  // 计算每张卡片的起始帧（按 duration_sec * fps 累加）
  let frameCursor = 0;
  const cardTimings = desc.cards.map((card) => {
    const startFrame = frameCursor;
    const durFrames = Math.round(card.duration_sec * desc.fps);
    frameCursor += durFrames;
    // 当前 card 在 expression 列表中的位置（1-based），非 expression card 为 0
    const exprIdx = card.type === 'expression'
      ? expressionCards.findIndex((c) => c.index === card.index) + 1
      : 0;
    return { card, startFrame, durFrames, exprIdx };
  });

  return (
    <AbsoluteFill style={{ background: THEME_COLORS[desc.theme].bg }}>
      {/* 各卡片 Sequence 拼接 */}
      {cardTimings.map(({ card, startFrame, durFrames, exprIdx }) => (
        <Sequence key={card.index} from={startFrame} durationInFrames={durFrames}>
          {card.type === 'intro' && (
            <IntroCard
              meta={desc.meta}
              tts_segments={card.tts_segments}
              theme={desc.theme}
              dayNumber={dayNumber}
              scene_image={desc.scene_image}
            />
          )}
          {card.type === 'expression' && (
            <ExpressionCard
              card={{
                index: card.index,
                style: card.style ?? 'neutral',
                literal_translation: card.literal_translation ?? '',
                sentence_en: card.sentence_en ?? '',
                phonetic: card.phonetic ?? '',
                note: card.note ?? '',
                tts_segments: card.tts_segments,
              }}
              theme={desc.theme}
              progress={expressionTotal > 0 ? { current: exprIdx, total: expressionTotal } : undefined}
            />
          )}
          {card.type === 'summary' && (
            <SummaryCard
              meta={desc.meta}
              tts_segments={card.tts_segments}
              theme={desc.theme}
            />
          )}
        </Sequence>
      ))}

      {/* 全局 Header / Footer（盖在最上层，全程可见） */}
      <Header text={headerText ?? '蛙蛙英语口语 | 每日一句'} theme={desc.theme} />
      <Footer text={footerText ?? '@en-sentence-study'} theme={desc.theme} />
    </AbsoluteFill>
  );
};