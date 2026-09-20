/**
 * 极简补间动画工具：只做数值插值，用来实现相机 flyTo 等平滑动画。
 */

export type EaseFn = (t: number) => number

export const easeInOutCubic: EaseFn = (t) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2

interface TweenState {
  from: number
  to: number
  start: number
  duration: number
  ease: EaseFn
}

const tweens = new Map<symbol, { state: TweenState; update: (v: number) => void; done?: () => void }>()

/**
 * 启动一个补间；返回 token，可用 cancelTween 取消。
 */
export function startTween(
  from: number,
  to: number,
  durationMs: number,
  update: (value: number) => void,
  opts: { ease?: EaseFn; done?: () => void } = {},
): symbol {
  const token = Symbol('tween')
  tweens.set(token, {
    state: { from, to, start: performance.now(), duration: durationMs, ease: opts.ease ?? easeInOutCubic },
    update,
    done: opts.done,
  })
  return token
}

export function cancelTween(token: symbol | null): void {
  if (token) tweens.delete(token)
}

/** 每帧调用：推进所有活跃补间，返回是否还有活跃补间 */
export function updateTweens(): boolean {
  const now = performance.now()
  for (const [token, t] of tweens) {
    const k = Math.min(1, (now - t.state.start) / t.state.duration)
    t.update(t.state.from + (t.state.to - t.state.from) * t.state.ease(k))
    if (k >= 1) {
      tweens.delete(token)
      t.done?.()
    }
  }
  return tweens.size > 0
}
