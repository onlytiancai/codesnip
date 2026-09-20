/**
 * 恒星分类（OBAFGKM 简化版）。
 */

export type StarClass = 'O' | 'B' | 'A' | 'F' | 'G' | 'K' | 'M'

export interface StarClassInfo {
  cls: StarClass
  name: string
  color: [number, number, number]
  /** scene 单位半径 */
  radius: number
  weight: number
  shortName: string
}

export const STAR_CLASSES: StarClassInfo[] = [
  { cls: 'O', name: 'O 型蓝白炽星', color: [155, 176, 255], radius: 6.5, weight: 0.001, shortName: 'O' },
  { cls: 'B', name: 'B 型蓝白星', color: [170, 191, 255], radius: 4.2, weight: 0.01, shortName: 'B' },
  { cls: 'A', name: 'A 型白星', color: [220, 230, 255], radius: 2.4, weight: 0.04, shortName: 'A' },
  { cls: 'F', name: 'F 型黄白星', color: [255, 244, 230], radius: 1.6, weight: 0.08, shortName: 'F' },
  { cls: 'G', name: 'G 型黄星', color: [255, 232, 180], radius: 1.2, weight: 0.12, shortName: 'G' },
  { cls: 'K', name: 'K 型橙星', color: [255, 196, 130], radius: 0.9, weight: 0.2, shortName: 'K' },
  { cls: 'M', name: 'M 型红矮星', color: [255, 170, 110], radius: 0.5, weight: 0.55, shortName: 'M' },
]

const TOTAL_WEIGHT = STAR_CLASSES.reduce((s, c) => s + c.weight, 0)

export function sampleStarClass(rng: () => number): StarClassInfo {
  let r = rng() * TOTAL_WEIGHT
  for (const c of STAR_CLASSES) {
    r -= c.weight
    if (r < 0) return c
  }
  return STAR_CLASSES[STAR_CLASSES.length - 1]
}

export type PlanetType =
  | 'lava'
  | 'rocky'
  | 'water'
  | 'desert'
  | 'ice'
  | 'gasGiant'
  | 'iceGiant'
  | 'dwarf'

export const PLANET_TEXTURE: Record<PlanetType, string> = {
  lava: 'mercuryTexture',
  rocky: 'marsTexture',
  water: 'earthTexture',
  desert: 'venusTexture',
  ice: 'uranusTexture',
  gasGiant: 'jupiterTexture',
  iceGiant: 'neptuneTexture',
  dwarf: 'moonTexture',
}