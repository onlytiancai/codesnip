/**
 * splitmix32 — 种子化的伪随机数生成器。
 *
 * 周期 2³²，O(1) 状态，对坐标哈希友好。
 */

export type Rng32 = () => number

/** splitmix32 单步 → 返回新的 a（u32），使用方按需归一化到 [0, 1) */
export function splitmix32(seed: number): Rng32 {
  let a = seed >>> 0
  return () => {
    a = (a + 0x9e3779b9) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** 从父 rng 取一次当种子，得到一个独立的 splitmix32 */
export function forkRng(parent: Rng32): Rng32 {
  return splitmix32((parent() * 0xffffffff) >>> 0)
}