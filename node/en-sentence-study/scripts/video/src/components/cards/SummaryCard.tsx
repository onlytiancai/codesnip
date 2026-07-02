import { AbsoluteFill, Audio, Sequence, interpolate, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import { FONT_FAMILY, LAYOUT, THEME_COLORS, Theme, TtsSegment, toStaticFile } from '../../theme';

type Props = {
  meta: {
    explanation: string;
  };
  tts_segments: TtsSegment[];
  theme: Theme;
};

/**
 * 总结卡（card N+1）
 * 布局：SUMMARY eyebrow → 整段 explanation（按长度自适应字号）
 * 动效：eyebrow → text 淡入
 * 音频：单段 zh（整段 explanation）
 */
export const SummaryCard: React.FC<Props> = ({ meta, tts_segments, theme }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const c = THEME_COLORS[theme];

  const explanationLen = meta.explanation.length;
  // 长度自适应字号：< 200 → 64px，200~400 → 56px，400~600 → 48px，>= 600 → 42px
  const textSize =
    explanationLen < 200 ? 64 : explanationLen < 400 ? 56 : explanationLen < 600 ? 48 : 42;

  const entrance = interpolate(frame, [0, 24], [0, 1], { extrapolateRight: 'clamp' });
  const entranceY = interpolate(frame, [0, 24], [30, 0], { extrapolateRight: 'clamp' });
  const exit = interpolate(
    frame,
    [durationInFrames - 18, durationInFrames],
    [1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  const fadeIn = (start: number) =>
    interpolate(frame, [start, start + 18], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });

  const seg0 = tts_segments[0];

  return (
    // ⚠️ 不要 background: c.bg，会把全局背景图挡死（Step 9 教训）
    <AbsoluteFill style={{ fontFamily: FONT_FAMILY, opacity: exit }}>
      <div
        style={{
          position: 'absolute',
          top: LAYOUT.headerHeight + LAYOUT.paddingY,
          bottom: LAYOUT.footerHeight + LAYOUT.paddingY,
          left: LAYOUT.paddingX,
          right: LAYOUT.paddingX,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          opacity: entrance,
          transform: `translateY(${entranceY}px)`,
        }}
      >
        <div
          style={{
            fontSize: 44,
            color: c.accent,
            fontWeight: 700,
            letterSpacing: '4px',
            marginBottom: 60,
            opacity: fadeIn(0),
          }}
        >
          SUMMARY · 今日小结
        </div>
        <p
          style={{
            fontSize: textSize,
            color: c.text,
            lineHeight: 1.6,
            wordWrap: 'break-word',
            wordBreak: 'break-word',
            margin: 0,
            opacity: fadeIn(12),
          }}
        >
          {meta.explanation}
        </p>
      </div>

      {/* 单段 zh 音频（包 Sequence 保持 API 一致性） */}
      {seg0 && (
        <Sequence from={0} durationInFrames={Math.round(seg0.duration_ms / 1000 * 30)}>
          <Audio src={staticFile(toStaticFile(seg0.audio_path))} />
        </Sequence>
      )}
    </AbsoluteFill>
  );
};