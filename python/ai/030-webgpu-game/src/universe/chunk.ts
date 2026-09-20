/**
 * Chunk 坐标系 + 种子派生。
 *
 * Chunk 是程序化宇宙的最小生成单元。
 * 飞船世界坐标 (px, py, pz) → 当前 chunk 索引 = floor(coord / CHUNK_SIZE)。
 * 每个 chunk 含恒星、行星、小行星带、彗星，按 chunk 种子确定性生成。
 *
 * 世界坐标用 BigInt 存储（避免 Float32 精度抖动）。
 */

import { splitmix32 } from './prng'

export const CHUNK_SIZE = 1000 // scene 单位（每个 chunk 是 1000³ box）

/** 主宇宙种子 —— 改这个值 = 换完整宇宙 */
export const UNIVERSE_SEED = 0xc0ffee42n

/** 32-bit hash 把 3D chunk 坐标混合成 u32 种子 */
function hashChunk(x: number, y: number, z: number): number {
  let h = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
  h = ((h ^ (h >>> 16)) * 0x85ebca6b) >>> 0
  h = ((h ^ (h >>> 13)) * 0xc2b2ae35) >>> 0
  return (h ^ (h >>> 16)) >>> 0
}

/** 给定 chunk 坐标，返回 splitmix32 随机数生成器 */
export function chunkRng(x: number, y: number, z: number): () => number {
  const h = hashChunk(x, y, z) ^ Number(UNIVERSE_SEED & 0xffffffffn)
  return splitmix32(h)
}

/** 世界坐标 → chunk 索引 */
export function worldToChunk(v: number): number {
  return Math.floor(v / CHUNK_SIZE)
}

/** 三个轴的 worldToChunk */
export function worldPosToChunk(x: number, y: number, z: number): [number, number, number] {
  return [worldToChunk(x), worldToChunk(y), worldToChunk(z)]
}

/** chunk 世界原点（chunk 锚点） */
export function chunkOrigin(cx: number, cy: number, cz: number): [number, number, number] {
  return [cx * CHUNK_SIZE, cy * CHUNK_SIZE, cz * CHUNK_SIZE]
}

/** "传送到指定坐标" 用的字符串格式：x,y,z 浮点（支持 ±）*/
export function parseWorldCoord(text: string): [number, number, number] | null {
  const parts = text.trim().replace(/^\(|\)$/g, '').split(/[,\s]+/)
  if (parts.length !== 3) return null
  const nums = parts.map((p) => Number(p))
  if (nums.some((n) => Number.isNaN(n))) return null
  return [nums[0], nums[1], nums[2]]
}

export function worldCoordToString(x: number, y: number, z: number): string {
  return `(${x.toFixed(0)}, ${y.toFixed(0)}, ${z.toFixed(0)})`
}

/** 飞船周围激活半径（多少 chunk 一圈） */
export const ACTIVE_CHUNK_RADIUS = 2 // 5×5×5 = 125 chunks 持续激活