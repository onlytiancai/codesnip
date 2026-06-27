import { AbsoluteFill, useCurrentFrame, interpolate, useVideoConfig } from 'remotion';

export type FooterProps = {
  text: string;
  theme: 'mint' | 'sunny';
};

const COLORS = {
  mint:  { text: '#5A8475', bg: 'rgba(255, 255, 255, 0.85)', border: '#CFE9DC' },
  sunny: { text: '#9A7B53', bg: 'rgba(255, 255, 255, 0.85)', border: '#F6DDB5' },
};

/**
 * 底部 Footer：前 12 帧淡入 + 上移
 */
export const Footer: React.FC<FooterProps> = ({ text, theme }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const c = COLORS[theme];

  // 0~12 帧：opacity 0→1, y 20→0
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
          height: 90,
          transform: `translateY(${-y}px)`,
          opacity,
          background: c.bg,
          backdropFilter: 'blur(8px)',
          borderTop: `2px solid ${c.border}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: 'system-ui, -apple-system, sans-serif',
        }}
      >
        <span style={{ fontSize: 26, color: c.text, fontWeight: 500, letterSpacing: '1px' }}>
          {text}
        </span>
      </div>
    </AbsoluteFill>
  );
};