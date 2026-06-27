import { AbsoluteFill, Audio, Img, interpolate, Sequence, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import { FONT_FAMILY, LAYOUT, THEME_COLORS, Theme, TtsSegment, toStaticFile } from '../../theme';

type Props = {
  meta: {
    scene_zh: string;
    task_zh: string;
    sentence_zh: string;
  };
  tts_segments: TtsSegment[];
  theme: Theme;
  dayNumber: number;
  scene_image?: {
    prompt: string;
    url: string | null;
    local_path: string;
  };
};

/**
 * 场景引入卡（card 0）— 作为视频封面
 * 布局：上半 = 16:9 横版主题图（顶部 banner 圆角面板），下半 = Day N 徽章 + SCENE/TASK/sentence 文案
 * 动效：无入场（视频第 0 帧直接是图片作为封面）；保留淡出用于平滑过渡
 */
export const IntroCard: React.FC<Props> = ({ meta, tts_segments, theme, dayNumber, scene_image }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const c = THEME_COLORS[theme];

  // 仅保留淡出（最后 18 帧），不做入场
  const exit = interpolate(
    frame,
    [durationInFrames - 18, durationInFrames],
    [1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  const seg0 = tts_segments[0];

  // 横版主题图区域：宽度 90%（972px），水平居中，高度按 16:9 自适应（547px）
  const imageWidth = 1080 * 0.9; // 972px
  const imageLeft = (1080 - imageWidth) / 2; // 54px 居中
  const imageHeight = imageWidth / (16 / 9); // 547px
  const imageTop = LAYOUT.headerHeight + 40;
  const textTop = imageTop + imageHeight + 40;

  return (
    <AbsoluteFill style={{ fontFamily: FONT_FAMILY, opacity: exit }}>
      {/* ① 16:9 横版主题图（顶部 banner，宽 90%，水平居中，高自适应） */}
      {scene_image?.local_path && (
        <Img
          src={staticFile(toStaticFile(scene_image.local_path))}
          style={{
            position: 'absolute',
            top: imageTop,
            left: imageLeft,
            width: imageWidth,
            height: imageHeight,
            borderRadius: 32,
            boxShadow: '0 12px 40px rgba(14, 59, 46, 0.18)',
          }}
        />
      )}

      {/* ② 文案区（横版图下方到 footer 上方） */}
      <div
        style={{
          position: 'absolute',
          top: textTop,
          bottom: LAYOUT.footerHeight + 60,
          left: LAYOUT.paddingX,
          right: LAYOUT.paddingX,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'flex-start',
        }}
      >
        {/* Day N 徽章（从 desc.id 提取序号） */}
        <div
          style={{
            display: 'inline-block',
            padding: '12px 28px',
            borderRadius: 100,
            background: c.accent,
            color: '#FFFFFF',
            fontSize: 32,
            fontWeight: 700,
            letterSpacing: '2px',
            marginBottom: 28,
            boxShadow: `0 6px 20px ${c.accent}55`,
          }}
        >
          DAY {dayNumber}
        </div>

        <div
          style={{
            fontSize: 36,
            color: c.accent,
            fontWeight: 600,
            letterSpacing: '6px',
            marginBottom: 16,
          }}
        >
          SCENE
        </div>
        <h1
          style={{
            fontSize: 120,
            color: c.text,
            fontWeight: 800,
            margin: '0 0 24px 0',
            lineHeight: 1.05,
            wordWrap: 'break-word',
            wordBreak: 'break-word',
          }}
        >
          {meta.scene_zh}
        </h1>
        <div
          style={{
            width: 110,
            height: 6,
            background: c.accent,
            margin: '24px 0',
            borderRadius: 3,
          }}
        />
        <div
          style={{
            fontSize: 36,
            color: c.accent,
            fontWeight: 600,
            letterSpacing: '6px',
            marginBottom: 16,
          }}
        >
          TASK
        </div>
        <h2
          style={{
            fontSize: 88,
            color: c.accent,
            fontWeight: 700,
            margin: '0 0 36px 0',
            lineHeight: 1.15,
            wordWrap: 'break-word',
            wordBreak: 'break-word',
          }}
        >
          {meta.task_zh}
        </h2>
        <div
          style={{
            fontSize: 44,
            color: c.subtext,
            lineHeight: 1.4,
            wordWrap: 'break-word',
            wordBreak: 'break-word',
          }}
        >
          {meta.sentence_zh}
        </div>
      </div>

      {/* ③ 单段 zh 音频（包 Sequence 以保持 API 一致性） */}
      {seg0 && (
        <Sequence from={0} durationInFrames={Math.round(seg0.duration_ms / 1000 * 30)}>
          <Audio src={staticFile(toStaticFile(seg0.audio_path))} />
        </Sequence>
      )}
    </AbsoluteFill>
  );
};