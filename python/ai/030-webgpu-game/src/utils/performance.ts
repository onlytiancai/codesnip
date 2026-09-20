/**
 * 设备性能探测与质量分级。
 * 分级依据：硬件内存 / CPU 核心数 / 是否移动端，启动时一次性判定；
 * 运行时另做 3 秒 FPS 探测，掉帧严重则动态降 DPR。
 */

export type ResolvedQuality = 'high' | 'low'

const nav = navigator as Navigator & { deviceMemory?: number }

/** 硬件启发式：低内存（≤4GB）或核心数少（≤4）或移动端 → 低配 */
export function detectInitialQuality(): ResolvedQuality {
  const memory = nav.deviceMemory ?? 8
  const cores = navigator.hardwareConcurrency ?? 8
  const mobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent)
  if (memory <= 4 || cores <= 4 || mobile) return 'low'
  return 'high'
}

export interface QualityProfile {
  dprCap: number
  antialias: boolean
  textureWidth: number // 常规 2:1 贴图宽
  textureWidthLarge: number // 大行星（地球/木星/土星）贴图宽，近距离观察更清晰
  sphereSegments: number
  orbitPoints: number
  starCount: number
  nebulaCount: number
  bloomAllowed: boolean
  cloudLayer: boolean
}

const HIGH: QualityProfile = {
  dprCap: 2,
  antialias: true,
  textureWidth: 512,
  textureWidthLarge: 2048, // 大行星近距离放大时仍清晰（地球/木星/土星）
  sphereSegments: 96, // 高清段数：避免近距离下球壳呈多边形
  orbitPoints: 256,
  starCount: 5000,
  nebulaCount: 5,
  bloomAllowed: true,
  cloudLayer: true,
}

const LOW: QualityProfile = {
  dprCap: 1,
  antialias: false,
  textureWidth: 256,
  textureWidthLarge: 512,
  sphereSegments: 32,
  orbitPoints: 160,
  starCount: 1500,
  nebulaCount: 3,
  bloomAllowed: false,
  cloudLayer: false,
}

export function qualityProfile(q: ResolvedQuality): QualityProfile {
  return q === 'high' ? HIGH : LOW
}

/** 3 秒 FPS 探测：低于 24fps 返回 true（需要降级） */
export function probeLowFps(getFps: () => number, onVerdict: (slow: boolean) => void): void {
  const start = performance.now()
  const frames: number[] = []
  const id = setInterval(() => {
    frames.push(getFps())
  }, 500)
  setTimeout(() => {
    clearInterval(id)
    if (!frames.length) return
    const avg = frames.reduce((a, b) => a + b, 0) / frames.length
    onVerdict(avg < 24)
  }, 3000 + start - performance.now())
}
