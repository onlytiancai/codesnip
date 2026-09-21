/**
 * ChunkManager：维护飞船周围活跃 chunk 集合。
 *
 * - 飞船移动时，每 250ms 检查一次"应激活的 chunk 集合"，对比当前集合，加载新 chunk、卸载远 chunk
 * - chunk group 挂在 playerGroup 下，position = chunkAnchor (world) - playerGroup.position（Float32 安全）
 * - 不做 LOD（MVP 7 简化），所有活跃 chunk 相同细节
 */

import * as THREE from 'three/webgpu'
import { ACTIVE_CHUNK_RADIUS, chunkOrigin, worldPosToChunk } from '../universe/chunk'
import { generateChunk } from '../universe/generateChunk'
import { buildChunk, type ChunkHandle } from './buildChunk'

interface ChunkEntry {
  handle: ChunkHandle
  position: THREE.Vector3 // 在 playerGroup 内的局部位置（= chunkWorld - playerPos）
  cx: number
  cy: number
  cz: number
  lastSeen: number
  /** 该 chunk 内所有恒星的世界坐标 + id（按 chunk 锚点 + 局部 position 算出） */
  stars: Array<{ id: number; worldPos: THREE.Vector3 }>
}

function chunkKey(cx: number, cy: number, cz: number): string {
  return `${cx},${cy},${cz}`
}

export class ChunkManager {
  private playerGroup: THREE.Group
  private chunks = new Map<string, ChunkEntry>()
  private lastSyncMs = 0
  private syncIntervalMs = 250

  constructor(playerGroup: THREE.Group) {
    this.playerGroup = playerGroup
  }

  /** 每帧调用 */
  update(_dt: number, time: number): void {
    const now = performance.now()
    if (now - this.lastSyncMs < this.syncIntervalMs) {
      // 仍然更新已有 chunk 的动画
      for (const entry of this.chunks.values()) {
        entry.handle.update(_dt, time)
      }
      return
    }
    this.lastSyncMs = now
    this.syncActiveChunks()
    for (const entry of this.chunks.values()) {
      entry.handle.update(_dt, time)
    }
  }

  /** 计算当前飞船周围应激活的 chunk 集合 */
  private syncActiveChunks(): void {
    const px = this.playerGroup.position.x
    const py = this.playerGroup.position.y
    const pz = this.playerGroup.position.z
    const [cx0, cy0, cz0] = worldPosToChunk(px, py, pz)
    const r = ACTIVE_CHUNK_RADIUS
    const want = new Set<string>()
    for (let dx = -r; dx <= r; dx++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dz = -r; dz <= r; dz++) {
          want.add(chunkKey(cx0 + dx, cy0 + dy, cz0 + dz))
        }
      }
    }

    // 卸载不在 want 里的
    for (const [key, entry] of this.chunks) {
      if (!want.has(key)) {
        entry.handle.dispose()
        this.chunks.delete(key)
      }
    }

    // 加载 want 里还没的
    for (const key of want) {
      if (this.chunks.has(key)) continue
      const [cx, cy, cz] = key.split(',').map(Number) as [number, number, number]
      const data = generateChunk(cx, cy, cz)
      const handle = buildChunk(data)
      // chunk 在 playerGroup 内的局部位置
      const [ox, oy, oz] = chunkOrigin(cx, cy, cz)
      const localPos = new THREE.Vector3(ox - px, oy - py, oz - pz)
      handle.group.position.copy(localPos)
      this.playerGroup.add(handle.group)
      // 计算每颗恒星的世界坐标（player 当前位置 + chunk 局部位置 + star 局部偏移）
      const stars: Array<{ id: number; worldPos: THREE.Vector3 }> = []
      for (const star of data.stars) {
        stars.push({
          id: star.id,
          worldPos: new THREE.Vector3(
            px + (ox - px) + star.position[0],
            py + (oy - py) + star.position[1],
            pz + (oz - pz) + star.position[2],
          ),
        })
      }
      this.chunks.set(key, {
        handle,
        position: localPos,
        cx,
        cy,
        cz,
        lastSeen: performance.now(),
        stars,
      })
    }

    // 更新已有 chunk 的位置（跟随 player 平移）+ 同步恒星世界坐标
    for (const entry of this.chunks.values()) {
      const [ox, oy, oz] = chunkOrigin(entry.cx, entry.cy, entry.cz)
      const newLocalX = ox - px
      const newLocalY = oy - py
      const newLocalZ = oz - pz
      entry.handle.group.position.set(newLocalX, newLocalY, newLocalZ)
      entry.position.set(newLocalX, newLocalY, newLocalZ)
      // 重新计算该 chunk 内恒星的世界坐标（chunk 局部位置变了，但相对 chunk 原点的恒星位置不变）
      for (let si = 0; si < entry.stars.length; si++) {
        const s = entry.stars[si]
        // 恒星相对 chunk 锚点的位置 = 星 (chunkAnchor + starLocal) - playerPos = entry.position + starLocal
        // 即 entry.handle.group 子节点位置（星 local） → 但这里存的是 world 坐标
        // chunk 在 scene 世界的位置 = entry.position + playerGroup.position
        const chunkWorld = new THREE.Vector3(
          entry.position.x + px,
          entry.position.y + py,
          entry.position.z + pz,
        )
        // star local relative to chunk anchor
        const star = entry.handle.starsLocal?.[si]
        if (star) {
          s.worldPos.set(
            chunkWorld.x + star[0],
            chunkWorld.y + star[1],
            chunkWorld.z + star[2],
          )
        }
      }
    }
  }

  /** 强制设位置（传送时调用） */
  resync(): void {
    this.lastSyncMs = 0
    this.syncActiveChunks()
  }

  /** 当前激活的 chunk 数（用于调试 / HUD） */
  getActiveCount(): number {
    return this.chunks.size
  }

  /** 收集所有激活恒星（id + worldPos） */
  getAllStars(): Array<{ id: number; worldPos: THREE.Vector3 }> {
    const out: Array<{ id: number; worldPos: THREE.Vector3 }> = []
    for (const entry of this.chunks.values()) {
      for (const s of entry.stars) out.push(s)
    }
    return out
  }

  dispose(): void {
    for (const entry of this.chunks.values()) {
      entry.handle.dispose()
    }
    this.chunks.clear()
  }
}