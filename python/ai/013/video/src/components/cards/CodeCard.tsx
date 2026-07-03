import { AbsoluteFill, Audio, Sequence, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import {
  CardPayload,
  FONT_FAMILY,
  LAYOUT,
  MONO_FONT,
  THEME_COLORS,
  Theme,
  TtsSegment,
} from '../../theme';
import { useFadeIn, useFadeOut } from '../../hooks/useFadeIn';
import { RichText } from '../RichText';

type Props = {
  payload: CardPayload;
  tts_segments: TtsSegment[];
  theme: Theme;
};

/**
 * CodeCard — 等宽字体 + 行高亮 + 整体卡片
 */
export const CodeCard: React.FC<Props> = ({ payload, tts_segments, theme }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const c = THEME_COLORS[theme];

  const exit = useFadeOut(durationInFrames);
  const cardFade = useFadeIn(0);
  const captionFade = useFadeIn(48);

  const codeLines = (payload.code_text ?? '').split('\n');
  const highlightSet = new Set(payload.code_highlight_lines ?? []);

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
          opacity: cardFade,
        }}
      >
        {/* 大标题 */}
        {payload.headline && (
          <div
            style={{
              fontSize: 56,
              fontWeight: 700,
              color: c.accent,
              marginBottom: 36,
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

        {/* 代码面板 */}
        <div
          style={{
            width: '90%',
            maxWidth: 1500,
            background: c.codeBg,
            color: c.codeText,
            borderRadius: 16,
            padding: '36px 48px',
            fontFamily: MONO_FONT,
            fontSize: 28,
            lineHeight: 1.55,
            boxShadow: `0 12px 40px ${c.border}`,
            overflow: 'hidden',
          }}
        >
          {codeLines.map((line, idx) => {
            const isHighlighted = highlightSet.has(idx);
            // 偶数行内,高亮行用整行 highlight
            return (
              <div
                key={idx}
                style={{
                  paddingLeft: 16,
                  paddingRight: 16,
                  background: isHighlighted ? c.codeLineHighlight : 'transparent',
                  borderLeft: `4px solid ${isHighlighted ? c.accent : 'transparent'}`,
                  color: isHighlighted ? '#FFE7CC' : c.codeText,
                  fontWeight: isHighlighted ? 700 : 400,
                  marginLeft: -16,
                  marginRight: -48,
                  paddingTop: 4,
                  paddingBottom: 4,
                }}
              >
                <span style={{ opacity: 0.45, marginRight: 18, userSelect: 'none' }}>
                  {String(idx + 1).padStart(2, '0')}
                </span>
                <span>{line || ' '}</span>
              </div>
            );
          })}
        </div>

        {/* Caption */}
        {payload.caption && (
          <div
            style={{
              marginTop: 32,
              padding: '14px 32px',
              background: c.highlight,
              color: c.accent,
              fontSize: 32,
              fontWeight: 600,
              borderRadius: 12,
              opacity: captionFade,
              letterSpacing: '1px',
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

      {tts_segments.map((seg, i) => {
        const startFrame = 0;
        const dur = Math.max(1, Math.round((seg.duration_ms / 1000) * fps));
        return (
          <Sequence key={i} from={startFrame} durationInFrames={dur}>
            <Audio src={staticFile(seg.audio_path)} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
