import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion';
import { FONT_FAMILY, THEME_COLORS, Theme } from '../theme';

export type FooterProps = {
  text: string;
  theme: Theme;
};

export const Footer: React.FC<FooterProps> = ({ text, theme }) => {
  const frame = useCurrentFrame();
  const c = THEME_COLORS[theme];

  const opacity = interpolate(frame, [0, 12], [0, 1], { extrapolateRight: 'clamp' });
  const y = interpolate(frame, [0, 12], [20, 0], { extrapolateRight: 'clamp' });

  return (
    <AbsoluteFill style={{ pointerEvents: 'none' }}>
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: 70,
          transform: `translateY(${-y}px)`,
          opacity,
          background: 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(8px)',
          borderTop: `2px solid ${c.border}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: FONT_FAMILY,
        }}
      >
        <span style={{ fontSize: 22, color: c.subtext, fontWeight: 500, letterSpacing: '1px' }}>
          {text}
        </span>
      </div>
    </AbsoluteFill>
  );
};
