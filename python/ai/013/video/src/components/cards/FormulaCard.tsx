import { AbsoluteFill, Audio, Img, Sequence, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import {
  assetToStaticFile,
  CardPayload,
  FONT_FAMILY,
  FrameStep,
  LatexStep,
  LAYOUT,
  THEME_COLORS,
  Theme,
  TtsSegment,
} from '../../theme';
import { useFadeIn, useFadeOut } from '../../hooks/useFadeIn';
import { MathLatex } from '../MathLatex';
import { RichText } from '../RichText';

type Props = {
  payload: CardPayload;
  tts_segments: TtsSegment[];
  theme: Theme;
};

/**
 * FormulaCard — 覆盖 formula / math_anim / plot / diagram 四种 type:
 *  - 单图模式: <Img> payload.image
 *  - math_anim PNG 模式 (payload.steps 存在 + 全是 image): 序列帧 cross-fade
 *  - math_anim KaTeX 模式 (payload.latex_steps 存在): KaTeX 实时渲染,逐行淡入
 */
export const FormulaCard: React.FC<Props> = ({ payload, tts_segments, theme }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const c = THEME_COLORS[theme];

  const exit = useFadeOut(durationInFrames);
  const hasPngSteps = (payload.steps?.length ?? 0) > 0;
  const hasLatexSteps = (payload.latex_steps?.length ?? 0) > 0;
  const hasSteps = hasPngSteps || hasLatexSteps;

  // 单图模式的整体淡入
  const fade0 = useFadeIn(0);
  const captionFadeIn = useFadeIn(36);

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
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {/* Headline(可选,大标题) */}
        {payload.headline && (
          <div
            style={{
              fontSize: 56,
              fontWeight: 700,
              color: c.accent,
              marginBottom: 36,
              opacity: useFadeIn(0),
              letterSpacing: '2px',
            }}
          >
            <RichText
              input={payload.headline ?? ''}
              fontSize={56}
              color={c.accent}
              inlineScale={0.7}
              fontWeight={700}
              lineHeight={1.1}
              style={{ letterSpacing: '2px' }}
            />
          </div>
        )}

        {/* Image area */}
        <div
          style={{
            flex: 1,
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            opacity: fade0,
          }}
        >
          {hasLatexSteps ? (
            <AnimLatexSteps steps={payload.latex_steps!} accent={c.accent} />
          ) : hasPngSteps ? (
            <AnimPngSteps steps={payload.steps!} />
          ) : payload.image ? (
            <Img
              src={staticFile(assetToStaticFile(payload.image))}
              style={{
                maxWidth: '100%',
                maxHeight: '100%',
                objectFit: 'contain',
              }}
            />
          ) : null}
        </div>

        {/* Caption(bottom subtitle) */}
        {payload.caption && (
          <div
            style={{
              marginTop: 24,
              padding: '14px 32px',
              background: c.highlight,
              color: c.accent,
              fontSize: 32,
              fontWeight: 600,
              borderRadius: 12,
              opacity: captionFadeIn,
              letterSpacing: '1px',
              textAlign: 'center',
            }}
          >
            » <RichText
              input={payload.caption ?? ''}
              fontSize={32}
              color={c.accent}
              inlineScale={0.7}
              fontWeight={600}
              lineHeight={1.3}
              style={{ letterSpacing: '1px' }}
            />
          </div>
        )}
      </div>
      {/* 音频 */}
      {tts_segments.map((seg, i) => {
        const dur = Math.max(1, Math.round((seg.duration_ms / 1000) * fps));
        return (
          <Sequence
            key={i}
            from={0}
            durationInFrames={dur}
            style={{
              translate: "-138.1px 1.2px"
            }}>
            <Audio src={staticFile(seg.audio_path)} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

/**
 * AnimPngSteps — 5 张 PNG 在连续区间内做交叉淡入(原实现,保留以向后兼容)
 */
const AnimPngSteps: React.FC<{ steps: FrameStep[] }> = ({ steps }) => {
  const frame = useCurrentFrame();
  if (steps.length === 0) return null;
  return (
    <>
      {steps.map((s, i) => {
        const next = steps[i + 1];
        const fadeOutEnd = next ? next.start_frame : s.end_frame;

        const fadeIn = interpolateStep(frame, [s.start_frame, s.start_frame + 15], [0, 1]);
        const fadeOut = next
          ? interpolateStep(frame, [fadeOutEnd - 15, fadeOutEnd], [1, 0])
          : 1;
        const opacity = Math.max(0, Math.min(1, fadeIn * fadeOut));

        return (
          <Img
            key={i}
            src={staticFile(assetToStaticFile(s.image))}
            style={{
              position: 'absolute',
              inset: 0,
              margin: 'auto',
              maxWidth: '100%',
              maxHeight: '100%',
              objectFit: 'contain',
              opacity,
            }}
          />
        );
      })}
    </>
  );
};

/**
 * AnimLatexSteps — KaTeX 实时渲染版本:
 *   - 所有步骤同时 mount,各自独立 opacity
 *   - 当前步骤(start_frame..end_frame)全不透明,上下区间淡入淡出
 *   - 每一步首次进入屏幕时整体上移 8px
 */
const AnimLatexSteps: React.FC<{ steps: LatexStep[]; accent: string }> = ({ steps, accent }) => {
  const frame = useCurrentFrame();
  return (
    <>
      {steps.map((s, i) => {
        const next = steps[i + 1];
        const fadeOutEnd = next ? next.start_frame : s.end_frame;

        const fadeIn = interpolateStep(frame, [s.start_frame, s.start_frame + 15], [0, 1]);
        const fadeOut = next
          ? interpolateStep(frame, [fadeOutEnd - 15, fadeOutEnd], [1, 0])
          : 1;
        const opacity = Math.max(0, Math.min(1, fadeIn * fadeOut));

        // 当前步骤整行加框;其它步骤仅基础 opacity
        const isCurrentStep = frame >= s.start_frame && frame < (next ? next.start_frame : s.end_frame);
        const isLastStep = i === steps.length - 1;
        const showBox = isCurrentStep || isLastStep;

        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              inset: 0,
              margin: 'auto',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              opacity,
            }}
          >
            <div
              style={{
                padding: showBox ? '24px 56px' : '14px 32px',
                border: showBox ? `3px solid ${accent}` : '3px solid transparent',
                background: showBox ? '#FFE7CC' : 'transparent',
                borderRadius: 18,
                transition: 'border-color 0.05s linear',
              }}
            >
              <MathLatex
                latex={s.latex}
                fontSize={showBox ? 76 : 64}
              />
            </div>
            {s.caption && (
              <div
                style={{
                  marginTop: 18,
                  fontFamily:
                    '"PingFang SC", "Helvetica Neue", system-ui, -apple-system, sans-serif',
                  fontSize: 28,
                  color: '#5C5C5C',
                  opacity: isCurrentStep ? 1 : 0.6,
                }}
              >
                {s.caption}
              </div>
            )}
          </div>
        );
      })}
    </>
  );
};

function interpolateStep(
  frame: number,
  input: [number, number],
  output: [number, number],
): number {
  const [a, b] = input;
  const [x, y] = output;
  if (b === a) return y;
  if (frame <= a) return x;
  if (frame >= b) return y;
  const t = (frame - a) / (b - a);
  return x + (y - x) * t;
}
