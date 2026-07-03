import { AbsoluteFill, Audio, Sequence, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import {
  assetToStaticFile,
  CardPayload,
  FONT_FAMILY,
  LAYOUT,
  THEME_COLORS,
  Theme,
  TtsSegment,
} from '../../theme';
import { useFadeIn, useFadeOut } from '../../hooks/useFadeIn';
import { RichText } from '../RichText';

type Props = {
  payload: CardPayload;        // headline + body
  tts_segments: TtsSegment[];
  theme: Theme;
};

export const IntroCard: React.FC<Props> = ({ payload, tts_segments, theme }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const c = THEME_COLORS[theme];

  // 整体淡入 + 上移
  const entrance = useFadeIn(0);
  const slideUp = interpolateFallback(frame, [0, 24], [40, 0]);
  const exit = useFadeOut(durationInFrames);

  // 逐行 fadeIn(body 每行 6 帧一格)
  const lines = (payload.body ?? '').split('\n');
  const lineOpacities = lines.map((_, i) =>
    interpolateFallback(frame, [12 + i * 6, 12 + i * 6 + 18], [0, 1])
  );

  const seg0 = tts_segments[0];
  const seg0EndFrame = seg0
    ? Math.round((seg0.duration_ms / 1000) * fps)
    : 0;

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
          justifyContent: 'center',
          opacity: entrance,
          transform: `translateY(${slideUp}px)`,
        }}
      >
        {/* 大标题(暖橙,公式风) */}
        <div
          style={{
            fontSize: 96,
            fontWeight: 800,
            color: c.text,
            lineHeight: 1.1,
            marginBottom: 48,
          }}
        >
          <RichText
            input={payload.headline ?? ''}
            fontSize={96}
            color={c.text}
            inlineScale={0.62}
            lineHeight={1.1}
            fontWeight={800}
          />
        </div>

        {/* 水平 accent bar */}
        <div
          style={{
            width: 160,
            height: 8,
            background: c.accent,
            borderRadius: 4,
            marginBottom: 56,
          }}
        />

        {/* 段落(body 按行 fadeIn) */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {lines.map((line, i) => (
            <div
              key={i}
              style={{
                fontSize: 38,
                color: i === 0 ? c.text : c.subtext,
                lineHeight: 1.45,
                opacity: lineOpacities[i],
                fontWeight: i === 0 ? 600 : 400,
              }}
            >
              <RichText
                input={line}
                fontSize={i === 0 ? 38 : 36}
                color={i === 0 ? c.text : c.subtext}
                inlineScale={0.65}
                fontWeight={i === 0 ? 600 : 400}
                lineHeight={1.45}
              />
            </div>
          ))}
        </div>

        {/* 字幕(caption,与 seg0 段尾同步) */}
        {payload.caption && (
          <div
            style={{
              position: 'absolute',
              bottom: LAYOUT.paddingY + 16,
              left: 0,
              right: 0,
              opacity: interpolateFallback(frame, [Math.max(0, seg0EndFrame - 60), seg0EndFrame], [0, 1]),
              textAlign: 'center',
              fontSize: 36,
              color: c.accent,
              fontWeight: 600,
              letterSpacing: '1px',
            }}
          >
            » <RichText
              input={payload.caption ?? ''}
              fontSize={36}
              color={c.accent}
              inlineScale={0.7}
              fontWeight={600}
              style={{ letterSpacing: '1px' }}
            />
          </div>
        )}
      </div>

      {seg0 && (
        <Sequence from={0} durationInFrames={Math.round(seg0.duration_ms / 1000 * fps) || 1}>
          <Audio src={staticFile(seg0.audio_path)} />
        </Sequence>
      )}
    </AbsoluteFill>
  );
};

/** 小工具:内联 interpolate,避免被外部误命名 */
function interpolateFallback(
  frame: number,
  input: [number, number],
  output: [number, number],
) {
  const [a, b] = input;
  const [x, y] = output;
  if (frame <= a) return x;
  if (frame >= b) return y;
  const t = (frame - a) / (b - a);
  return x + (y - x) * t;
}
