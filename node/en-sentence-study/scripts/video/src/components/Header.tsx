import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';

export type HeaderProps = {
  text: string;
  theme: 'mint' | 'sunny';
};

const COLORS = {
  mint:  { text: '#0E3B2E', bg: 'rgba(255, 255, 255, 0.85)', border: '#CFE9DC', accent: '#19A974' },
  sunny: { text: '#3E2A14', bg: 'rgba(255, 255, 255, 0.85)', border: '#F6DDB5', accent: '#F58A1F' },
};

/**
 * 顶部 Header：从顶部滑入，前 12 帧动画
 */
export const Header: React.FC<HeaderProps> = ({ text, theme }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const c = COLORS[theme];

  // spring 从 y=-110 → y=0
  const slideIn = spring({ frame, fps, config: { damping: 14 } });
  const y = interpolate(slideIn, [0, 1], [-110, 0]);

  return (
    <AbsoluteFill style={{ pointerEvents: 'none' }}>
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: 110,
          transform: `translateY(${y}px)`,
          background: c.bg,
          backdropFilter: 'blur(8px)',
          borderBottom: `2px solid ${c.border}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: 'system-ui, -apple-system, sans-serif',
        }}
      >
        <div
          style={{
            width: 12,
            height: 36,
            background: c.accent,
            borderRadius: 6,
            marginRight: 16,
          }}
        />
        <span style={{ fontSize: 38, color: c.text, fontWeight: 700, letterSpacing: '2px' }}>
          {text}
        </span>
      </div>
    </AbsoluteFill>
  );
};