import { AbsoluteFill, Audio, Sequence, interpolate, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import {
  FONT_FAMILY,
  LAYOUT,
  PAUSE_FRAMES_AT_30FPS,
  STYLE_COLORS,
  StyleKind,
  THEME_COLORS,
  Theme,
  TtsSegment,
  toStaticFile,
} from '../../theme';

type Props = {
  card: {
    index: number;
    style: StyleKind;
    literal_translation: string;
    sentence_en: string;
    phonetic: string;
    note: string;
    tts_segments: TtsSegment[];
  };
  theme: Theme;
};

/**
 * 表达卡（card 1..N）
 * 布局：style badge → 中文 → 英文 → 音标 → note
 * 动效：badge/zh/note 前段淡入；en/phonetic 在 zh 段播完后淡入
 * 音频：2 段（先 zh 后 en）
 */
export const ExpressionCard: React.FC<Props> = ({ card, theme }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const c = THEME_COLORS[theme];
  const sc = STYLE_COLORS[card.style] ?? STYLE_COLORS.neutral;

  // 段起始帧（含段间停顿）：zh → PAUSE → en
  // PAUSE_FRAMES_AT_30FPS = 21，fps 不一致时按比例换算
  const pauseFrames = Math.round((PAUSE_FRAMES_AT_30FPS * fps) / 30);
  const segStartFrames = card.tts_segments.map((_seg, i) => {
    let acc = 0;
    for (let k = 0; k < i; k++) {
      acc += Math.round((card.tts_segments[k].duration_ms / 1000) * fps);
      acc += pauseFrames; // 每段之间都加停顿
    }
    return acc;
  });

  // en 段起始帧（= zh 段结束 + 停顿）
  const enStartFrame = segStartFrames[1] ?? 0;

  // 整体淡入 + 上移（前 24 帧）
  const entrance = interpolate(frame, [0, 24], [0, 1], { extrapolateRight: 'clamp' });
  const entranceY = interpolate(frame, [0, 24], [30, 0], { extrapolateRight: 'clamp' });
  // 整体淡出（最后 18 帧）
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
        {/* 1) Style 徽章 */}
        <div
          style={{
            display: 'inline-block',
            padding: '16px 36px',
            borderRadius: 100,
            fontSize: 36,
            fontWeight: 700,
            letterSpacing: '2px',
            marginBottom: 80,
            width: 'fit-content',
            background: sc.bg,
            color: sc.text,
            opacity: fadeIn(0),
          }}
        >
          {sc.label}
        </div>

        {/* 2) 中文（zh 段播放期间显示） */}
        <div
          style={{
            fontSize: 92,
            fontWeight: 600,
            color: c.text,
            lineHeight: 1.25,
            marginBottom: 50,
            wordWrap: 'break-word',
            wordBreak: 'break-word',
            opacity: fadeIn(6),
          }}
        >
          {card.literal_translation}
        </div>

        {/* 3) 英文（zh 段播完 + 停顿后才出现） */}
        <div
          style={{
            fontSize: 76,
            fontWeight: 600,
            color: c.accent,
            lineHeight: 1.25,
            marginBottom: 40,
            wordWrap: 'break-word',
            wordBreak: 'break-word',
            opacity: fadeIn(enStartFrame),
          }}
        >
          {card.sentence_en}
        </div>

        {/* 4) 音标（跟英文一起出现） */}
        <div
          style={{
            fontSize: 44,
            color: c.subtext,
            fontStyle: 'italic',
            marginBottom: 60,
            wordWrap: 'break-word',
            opacity: fadeIn(enStartFrame + 6),
          }}
        >
          /{card.phonetic}/
        </div>

        {/* 5) Note（淡入后一直显示，绿色左色条） */}
        <div
          style={{
            fontSize: 42,
            color: c.subtext,
            lineHeight: 1.5,
            padding: '30px 40px',
            background: c.noteBg,
            borderRadius: 24,
            borderLeft: `8px solid ${c.accent}`,
            wordWrap: 'break-word',
            opacity: fadeIn(12),
          }}
        >
          {card.note}
        </div>
      </div>

      {/* 2 段音频（先 zh 后 en，段间 PAUSE 停顿；每段包 Sequence 避免重叠） */}
      {card.tts_segments.map((seg, i) => {
        const startFrame = segStartFrames[i];
        const durFrames = Math.round((seg.duration_ms / 1000) * fps);
        return (
          <Sequence key={i} from={startFrame} durationInFrames={durFrames}>
            <Audio src={staticFile(toStaticFile(seg.audio_path))} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};