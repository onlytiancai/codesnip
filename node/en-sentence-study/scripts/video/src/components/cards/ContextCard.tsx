import { AbsoluteFill, Audio, Sequence, interpolate, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import { FONT_FAMILY, LAYOUT, THEME_COLORS, Theme, TtsSegment, toStaticFile } from '../../theme';

type Props = {
  meta: {
    scene_zh: string;
    task_zh: string;
    sentence_zh: string;
    context: string;
  };
  tts_segments: TtsSegment[];
  theme: Theme;
  dayNumber: number;
};

/**
 * 场景上下文卡（紧跟 1s 封面之后）
 * 布局：上半 = DAY N 徽章 + SCENE / TASK 紧凑 meta 横排
 *       下半 = context 叙事正文（配 0-0 mp3 单段音频）
 * 动效：整块 fadeIn + translateY 入场，最后 18 帧淡出
 */
export const ContextCard: React.FC<Props> = ({ meta, tts_segments, theme, dayNumber }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const c = THEME_COLORS[theme];

  const seg0 = tts_segments[0];
  // 优先用 tts 段的文本（与音频对齐），缺则回落 meta.context
  const contextText = seg0?.text ?? meta.context ?? '';

  const entrance = interpolate(frame, [0, 18], [0, 1], { extrapolateRight: 'clamp' });
  const entranceY = interpolate(frame, [0, 18], [20, 0], { extrapolateRight: 'clamp' });
  const exit = interpolate(
    frame,
    [durationInFrames - 18, durationInFrames],
    [1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  return (
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
          opacity: entrance,
          transform: `translateY(${entranceY}px)`,
        }}
      >
        {/* DAY N 徽章 */}
        <div
          style={{
            display: 'inline-block',
            padding: '10px 22px',
            borderRadius: 100,
            background: c.accent,
            color: '#FFFFFF',
            fontSize: 28,
            fontWeight: 700,
            letterSpacing: '2px',
            marginBottom: 20,
            boxShadow: `0 6px 18px ${c.accent}55`,
            alignSelf: 'flex-start',
          }}
        >
          DAY {dayNumber}
        </div>

        {/* SCENE 行 */}
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, marginBottom: 10 }}>
          <div
            style={{
              fontSize: 30,
              color: c.accent,
              fontWeight: 600,
              letterSpacing: '4px',
            }}
          >
            SCENE
          </div>
          <div style={{ fontSize: 48, color: c.text, fontWeight: 700, lineHeight: 1.1 }}>
            {meta.scene_zh}
          </div>
        </div>

        {/* TASK 行 */}
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, marginBottom: 28 }}>
          <div
            style={{
              fontSize: 30,
              color: c.accent,
              fontWeight: 600,
              letterSpacing: '4px',
            }}
          >
            TASK
          </div>
          <div style={{ fontSize: 40, color: c.accent, fontWeight: 700, lineHeight: 1.2 }}>
            {meta.task_zh}
          </div>
        </div>

        {/* 分隔线 */}
        <div
          style={{
            width: 100,
            height: 4,
            background: c.accent,
            margin: '0 0 32px 0',
            borderRadius: 2,
          }}
        />

        {/* context 叙事正文 */}
        <p
          style={{
            fontSize: 46,
            color: c.text,
            lineHeight: 1.7,
            wordWrap: 'break-word',
            wordBreak: 'break-word',
            margin: 0,
          }}
        >
          {contextText}
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