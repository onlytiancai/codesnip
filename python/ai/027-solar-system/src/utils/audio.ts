/**
 * 极简 WebAudio 合成音效（无音频文件，默认关闭，用户主动开启才播放）。
 */

let ctx: AudioContext | null = null
let enabled = false

export function setSoundEnabled(v: boolean): void {
  enabled = v
  if (!v) return
  // 首次开启时惰性创建 AudioContext（必须在用户手势中）
  if (!ctx) {
    const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (AC) ctx = new AC()
  }
  void ctx?.resume()
}

function tone(freq: number, duration: number, type: OscillatorType = 'sine', gain = 0.06, when = 0): void {
  if (!enabled || !ctx) return
  const t0 = ctx.currentTime + when
  const osc = ctx.createOscillator()
  const g = ctx.createGain()
  osc.type = type
  osc.frequency.value = freq
  g.gain.setValueAtTime(gain, t0)
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + duration)
  osc.connect(g).connect(ctx.destination)
  osc.start(t0)
  osc.stop(t0 + duration + 0.02)
}

/** UI 按钮点击 */
export function playClick(): void {
  tone(660, 0.06, 'sine', 0.05)
}

/** 选中天体：短促双音 */
export function playSelect(): void {
  tone(523, 0.09, 'sine', 0.06)
  tone(784, 0.12, 'sine', 0.06, 0.07)
}

/** 关闭面板 */
export function playClose(): void {
  tone(392, 0.08, 'sine', 0.05)
}
