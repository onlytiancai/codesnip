import { AbsoluteFill, Sequence } from 'remotion';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { IntroCard } from './components/cards/IntroCard';
import { FormulaCard } from './components/cards/FormulaCard';
import { CodeCard } from './components/cards/CodeCard';
import { DescJson, THEME_COLORS } from './theme';

export type { DescJson };

export type VideoProps = {
  desc: DescJson;
  headerText?: string;
  footerText?: string;
};

function resolveDesc(props: VideoProps): DescJson {
  if ('cards' in props && 'meta' in props) {
    return props as unknown as DescJson;
  }
  return (props as { desc: DescJson }).desc;
}

/**
 * 端到端视频主组合:
 *  - 1920×1080 paper 主题
 *  - 每张 card 用 <Sequence> 拼接到正确时间区间
 *  - 3 路分发: intro / (formula+math_anim+plot+diagram) / code
 */
export const Video: React.FC<VideoProps> = (props) => {
  const desc = resolveDesc(props);
  const { headerText, footerText } = props;
  const fps = desc.fps;

  // 计算每张卡片的起始帧 (按 duration_sec * fps 累加)
  let frameCursor = 0;
  const cardTimings = desc.cards.map((card) => {
    const startFrame = frameCursor;
    const durFrames = Math.round(card.duration_sec * fps);
    frameCursor += durFrames;
    return { card, startFrame, durFrames };
  });

  return (
    <AbsoluteFill style={{ background: THEME_COLORS[desc.theme].bg }}>
      {/* 各卡片 Sequence 拼接 */}
      {cardTimings.map(({ card, startFrame, durFrames }) => (
        <Sequence key={card.index} from={startFrame} durationInFrames={durFrames}>
          {card.type === 'intro' && (
            <IntroCard
              payload={card.payload ?? {}}
              tts_segments={card.tts_segments}
              theme={desc.theme}
            />
          )}
          {(card.type === 'formula' ||
            card.type === 'math_anim' ||
            card.type === 'plot' ||
            card.type === 'diagram') && (
            <FormulaCard
              payload={card.payload ?? {}}
              tts_segments={card.tts_segments}
              theme={desc.theme}
            />
          )}
          {card.type === 'code' && (
            <CodeCard
              payload={card.payload ?? {}}
              tts_segments={card.tts_segments}
              theme={desc.theme}
            />
          )}
        </Sequence>
      ))}

      {/* 全局 Header / Footer */}
      <Header text={headerText ?? desc.meta.title} theme={desc.theme} />
      <Footer text={footerText ?? '@xor-bp'} theme={desc.theme} />
    </AbsoluteFill>
  );
};
