import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion';
import { FONT_FAMILY, THEME_COLORS, Theme } from '../theme';

export type HeaderProps = {
  text: string;
  theme: Theme;
};

export const Header: React.FC<HeaderProps> = ({ text, theme }) => {
  const frame = useCurrentFrame();
  const c = THEME_COLORS[theme];

  // 前 12 帧:从顶部滑入
  const slideIn = interpolate(frame, [0, 12], [-90, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const opacity = interpolate(frame, [0, 12], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill style={{ pointerEvents: 'none' }}>
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: 90,
          transform: `translateY(${slideIn}px)`,
          opacity,
          background: 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(8px)',
          borderBottom: `2px solid ${c.border}`,
          display: 'flex',
          alignItems: 'center',
          paddingLeft: 56,
          fontFamily: FONT_FAMILY,
        }}
      >
        <div
          style={{
            width: 8,
            height: 36,
            background: c.accent,
            borderRadius: 4,
            marginRight: 24,
          }}
        />
        <span style={{ fontSize: 32, color: c.text, fontWeight: 700, letterSpacing: '2px' }}>
          {text}
        </span>
      </div>
    </AbsoluteFill>
  );
};
