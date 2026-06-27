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
  scene_image?: {
    prompt: string;
    url: string | null;
    local_path: string;
  };
};

/**
 * 场景引入卡（card 0）— 作为视频封面
 * 布局：上半 60% = 主题插画（圆角面板），下半 40% = SCENE/TASK/sentence 文案
 * 动效：无入场（视频第 0 帧直接是图片作为封面）；保留淡出用于平滑过渡
 */
export const IntroCard: React.FC<Props> = ({ meta, tts_segments, theme, scene_image }) => {
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

  // 主题图区域：top=headerHeight+60, height=1100（约 57% 屏幕）
  const imageTop = LAYOUT.headerHeight + 60;
  const imageHeight = 1100;

  return (
    <AbsoluteFill style={{ fontFamily: FONT_FAMILY, opacity: exit }}>
      {/* ① 主题图（上 57%，圆角面板，居中） */}
      {scene_image?.local_path && (
        <Img
          src={staticFile(toStaticFile(scene_image.local_path))}
          style={{
            position: 'absolute',
            top: imageTop,
            left: 60,
            right: 60,
            height: imageHeight,
            width: 'auto',
            objectFit: 'cover',
            borderRadius: 36,
            boxShadow: '0 12px 40px rgba(14, 59, 46, 0.18)',
          }}
        />
      )}

      {/* ② 文案区（图下方到 footer 上方，居中） */}
      <div
        style={{
          position: 'absolute',
          top: imageTop + imageHeight + 40,
          bottom: LAYOUT.footerHeight + 60,
          left: LAYOUT.paddingX,
          right: LAYOUT.paddingX,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'flex-start',
        }}
      >
        <div
          style={{
            fontSize: 32,
            color: c.accent,
            fontWeight: 600,
            letterSpacing: '5px',
            marginBottom: 12,
          }}
        >
          SCENE
        </div>
        <h1
          style={{
            fontSize: 96,
            color: c.text,
            fontWeight: 800,
            margin: '0 0 20px 0',
            lineHeight: 1.05,
            wordWrap: 'break-word',
            wordBreak: 'break-word',
          }}
        >
          {meta.scene_zh}
        </h1>
        <div
          style={{
            width: 100,
            height: 5,
            background: c.accent,
            margin: '20px 0',
            borderRadius: 3,
          }}
        />
        <div
          style={{
            fontSize: 32,
            color: c.accent,
            fontWeight: 600,
            letterSpacing: '5px',
            marginBottom: 12,
          }}
        >
          TASK
        </div>
        <h2
          style={{
            fontSize: 72,
            color: c.accent,
            fontWeight: 700,
            margin: '0 0 32px 0',
            lineHeight: 1.15,
            wordWrap: 'break-word',
            wordBreak: 'break-word',
          }}
        >
          {meta.task_zh}
        </h2>
        <div
          style={{
            fontSize: 40,
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