// 复用的淡入 hook + 上移/淡出辅助
import { interpolate, useCurrentFrame, useVideoConfig } from 'remotion';

export const FADE_DURATION = 18;

/**
 * 在 [start, start + 18] 帧之间做 0→1 透明度淡入,外侧 clamp。
 */
export const useFadeIn = (start: number): number => {
  const frame = useCurrentFrame();
  return interpolate(frame, [start, start + FADE_DURATION], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
};

/** 在 [end - 18, end] 之间做 1→0 淡出。 */
export const useFadeOut = (durationInFrames: number): number => {
  const frame = useCurrentFrame();
  return interpolate(
    frame,
    [durationInFrames - 18, durationInFrames],
    [1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
};

/** 在 [start, start + 18] 之间做 30px 上移 → 0。 */
export const useSlideUp = (start: number): number => {
  const frame = useCurrentFrame();
  return interpolate(frame, [start, start + FADE_DURATION], [30, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
};

/** 通用 frame param,常用模式:前 24 帧整体淡入 + 上移,后 18 帧整体淡出 */
export const useCardEntrance = (durationInFrames: number) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 24], [0, 1], { extrapolateRight: 'clamp' });
  const y = interpolate(frame, [0, 24], [30, 0], { extrapolateRight: 'clamp' });
  const exit = interpolate(
    frame,
    [durationInFrames - 18, durationInFrames],
    [1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
  return { opacity, y, exit };
};
