/**
 * 真实数据 → 视觉展示比例的换算层。
 *
 * 太阳系真实尺度（行星大小 vs 轨道距离）差异极大，无法 1:1 展示，
 * 因此这里采用「非线性压缩」：所有映射都基于真实数据推导，且全部
 * 集中在本文件，方便调整。
 */

import { BODIES } from '../data/planetData'

/** 地球视觉半径基准（scene 单位） */
export const EARTH_VISUAL_RADIUS = 1.0

/** 太阳视觉半径（真实为地球 109 倍，封顶展示） */
export const SUN_VISUAL_RADIUS = 4.0

/** 行星视觉半径下限（保证可点击） */
export const MIN_PLANET_RADIUS = 0.35

/**
 * 行星视觉半径：平方根压缩 + 下限。
 * 保持真实相对大小顺序：木星 > 土星 > 天王星 ≈ 海王星 > 地球 ≈ 金星 > 火星 > 水星
 */
export function visualRadius(radiusKm: number): number {
  const r = Math.sqrt(radiusKm / BODIES.earth.radiusKm)
  return Math.max(MIN_PLANET_RADIUS, r * EARTH_VISUAL_RADIUS)
}

// ---- 轨道距离：au^0.55 压缩后线性映射到 [6, 45] ----
const DIST_MIN_AU = 0.387 ** 0.55
const DIST_MAX_AU = 30.07 ** 0.55
const DIST_NEAR = 6 // 水星轨道视觉距离（> 太阳半径 4.0，避免被太阳吞没）
const DIST_FAR = 45 // 海王星轨道视觉距离

/** 轨道视觉距离：内行星密集、火木之间保留明显空隙、外行星更遥远 */
export function visualDistance(au: number): number {
  const s = au ** 0.55
  const t = (s - DIST_MIN_AU) / (DIST_MAX_AU - DIST_MIN_AU)
  return DIST_NEAR + t * (DIST_FAR - DIST_NEAR)
}

/** 月球绕地球的视觉距离（真实 60 倍地球半径，压缩展示） */
export const MOON_VISUAL_DISTANCE = 2.2

// ---- 公转周期：天数^0.6 压缩，水星 1x 速度下约 8 秒一圈 ----
/** 水星视觉公转周期（秒，1x 速度下） */
export const MERCURY_ORBIT_SECONDS = 8

/** 视觉公转周期（秒）：真实周期比例经过 0.6 次方压缩，水星依然明显最快 */
export function visualOrbitSeconds(realDays: number): number {
  return MERCURY_ORBIT_SECONDS * (realDays / BODIES.mercury.orbitalPeriodDays) ** 0.6
}

// ---- 自转：真实自转周期比例（小时）经平方根压缩 ----
const EARTH_SPIN_SECONDS = 2.0

/**
 * 视觉自转周期（秒）：木星最快、金星极慢。
 * 符号与真实数据一致（负值 = 逆向自转）。
 */
export function visualSpinSeconds(realHours: number): number {
  const sign = realHours < 0 ? -1 : 1
  return sign * EARTH_SPIN_SECONDS * Math.sqrt(Math.abs(realHours) / Math.abs(BODIES.earth.rotationPeriodHours))
}

/**
 * 时间速度档位（相对真实 1x；1x 已做过周期压缩，见 visualOrbitSeconds）
 */
export const SPEED_PRESETS = [0.1, 0.5, 1, 5, 10, 50, 100, 1000]
export const DEFAULT_TIME_SCALE = 1
