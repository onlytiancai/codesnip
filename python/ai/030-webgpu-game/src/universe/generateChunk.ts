/**
 * Chunk 程序化生成。
 *
 * 输入：(chunkX, chunkY, chunkZ)
 * 输出：ChunkData { stars[], planets[], belts[], comets[] }
 *
 * 位置约定：所有坐标是相对 chunk 锚点 (0,0,0) 的 scene 单位。
 * 渲染时 chunk group 平移到 (cx*CHUNK_SIZE, cy*CHUNK_SIZE, cz*CHUNK_SIZE)。
 *
 * 恒星位于 chunk 内 [-CHUNK_SIZE/2 + 50, CHUNK_SIZE/2 - 50] 范围；
 * 行星 / 彗星相对恒星位置。
 */

import { CHUNK_SIZE, chunkRng } from './chunk'
import { forkRng } from './prng'
import { sampleStarClass, type PlanetType, type StarClassInfo } from './starClasses'

export interface StarData {
  /** 相对 chunk 锚点 */
  position: [number, number, number]
  class: StarClassInfo
  radius: number
  /** 唯一编号（确定性：同一 chunk 同一恒星永远同一 id） */
  id: number
}

export interface PlanetData {
  /** 主星在 stars 数组里的 index */
  hostStarIndex: number
  /** 相对主星的距离（scene 单位） */
  distance: number
  type: PlanetType
  radius: number
  orbitPeriod: number
  orbitPhase: number
  spinPeriod: number
}

export interface CometData {
  hostStarIndex: number
  perihelion: number
  semiMajorAxis: number
  eccentricity: number
  orbitPeriod: number
  meanAnomaly0: number
  nucleusRadius: number
  tailSegments: number
}

export interface ChunkData {
  /** chunk 索引（用于定位 / HUD 显示） */
  cx: number
  cy: number
  cz: number
  stars: StarData[]
  planets: PlanetData[]
  comets: CometData[]
}

const MIN_STAR_DIST = 220
const HALF = CHUNK_SIZE / 2 - 80

function randomInChunk(rng: () => number, host?: [number, number, number]): [number, number, number] {
  // 围绕 host（或原点）在球壳上随机分布
  const cx = host ? host[0] : 0
  const cy = host ? host[1] : 0
  const cz = host ? host[2] : 0
  for (let i = 0; i < 16; i++) {
    const x = cx + (rng() - 0.5) * 2 * HALF
    const y = cy + (rng() - 0.5) * 2 * HALF
    const z = cz + (rng() - 0.5) * 2 * HALF
    if (Math.max(Math.abs(x), Math.abs(y), Math.abs(z)) > HALF) continue
    return [x, y, z]
  }
  // fallback: 在原点附近
  return [(rng() - 0.5) * HALF, (rng() - 0.5) * HALF, (rng() - 0.5) * HALF]
}

/** 解 Kepler 方程 E - e·sin(E) = M（牛顿迭代，硬限 5 次） */
export function solveKepler(M: number, e: number): number {
  let E = M
  for (let i = 0; i < 5; i++) {
    const dE = (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E))
    E -= dE
    if (Math.abs(dE) < 1e-6) break
  }
  return E
}

/** 把恒星合理分类（距离恒星的"半径倍数"决定行星类型） */
function classifyPlanetByDistance(distMul: number, rng: () => number): PlanetType {
  // distMul = distance / starRadius
  if (distMul < 5) return rng() < 0.5 ? 'lava' : 'rocky'
  if (distMul < 15) {
    const r = rng()
    if (r < 0.4) return 'rocky'
    if (r < 0.7) return 'water'
    if (r < 0.85) return 'desert'
    return 'ice'
  }
  if (distMul < 60) {
    return rng() < 0.65 ? 'gasGiant' : 'iceGiant'
  }
  return 'dwarf'
}

export function generateChunk(cx: number, cy: number, cz: number): ChunkData {
  const rootRng = chunkRng(cx, cy, cz)
  const positionRng = forkRng(rootRng)
  const starTypeRng = forkRng(rootRng)

  // ---------- 恒星（1~3 颗 / chunk） ----------
  const starCount = 1 + Math.floor(rootRng() * 3)
  const stars: StarData[] = []
  for (let i = 0; i < starCount; i++) {
    const cls = sampleStarClass(starTypeRng)
    let pos: [number, number, number] = [0, 0, 0]
    let attempts = 0
    while (attempts < 32) {
      pos = randomInChunk(positionRng, stars[0]?.position)
      let ok = true
      for (const s of stars) {
        const dx = pos[0] - s.position[0]
        const dy = pos[1] - s.position[1]
        const dz = pos[2] - s.position[2]
        if (Math.hypot(dx, dy, dz) < MIN_STAR_DIST) {
          ok = false
          break
        }
      }
      if (ok) break
      attempts++
    }
    // 唯一编号：u16（0~65535），种子化所以同坐标同恒星永远同一 id
    const id = Math.floor(starTypeRng() * 0x10000) >>> 0
    stars.push({ position: pos, class: cls, radius: cls.radius, id })
  }

  // ---------- 每颗恒星的行星 + 彗星 ----------
  const planets: PlanetData[] = []
  const comets: CometData[] = []

  for (let si = 0; si < stars.length; si++) {
    const star = stars[si]
    const orbitRng = forkRng(rootRng)
    const planetRng = forkRng(rootRng)
    const cometRng = forkRng(rootRng)

    // 行星：3~8 颗
    const pCount = 3 + Math.floor(orbitRng() * 6)
    const baseLog = Math.log(star.radius * 4)
    const maxLog = Math.log(star.radius * 80)
    const step = (maxLog - baseLog) / pCount
    for (let i = 0; i < pCount; i++) {
      const ideal = Math.exp(baseLog + step * (i + 0.5))
      const jitter = 1 + (planetRng() - 0.5) * 0.3
      const distance = ideal * jitter
      const type = classifyPlanetByDistance(distance / star.radius, planetRng)
      const baseRadius =
        type === 'gasGiant' ? 1.6 : type === 'iceGiant' ? 1.3 : type === 'dwarf' ? 0.25 : type === 'lava' ? 0.35 : 0.55
      const radius = baseRadius * (0.7 + planetRng() * 0.6)
      planets.push({
        hostStarIndex: si,
        distance,
        type,
        radius,
        orbitPeriod: Math.max(4, Math.pow(distance / star.radius, 1.5) * 2),
        orbitPhase: planetRng(),
        spinPeriod: 4 + planetRng() * 8,
      })
    }

    // 彗星（0~3）
    const cRoll = cometRng()
    const cCount = cRoll < 0.55 ? 0 : cRoll < 0.85 ? 1 : cRoll < 0.97 ? 2 : 3
    for (let i = 0; i < cCount; i++) {
      const e = 0.85 + cometRng() * 0.14
      const perihelion = (5 + cometRng() * 10) * star.radius
      const a = perihelion / (1 - e)
      comets.push({
        hostStarIndex: si,
        perihelion,
        semiMajorAxis: a,
        eccentricity: e,
        orbitPeriod: Math.pow(a / star.radius, 1.5) * 1.5,
        meanAnomaly0: cometRng() * Math.PI * 2,
        nucleusRadius: 0.04 + cometRng() * 0.06,
        tailSegments: 24,
      })
    }
  }

  return { cx, cy, cz, stars, planets, comets }
}