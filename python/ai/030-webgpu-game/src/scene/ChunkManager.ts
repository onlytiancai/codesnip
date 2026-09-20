/**
 * ChunkManager：维护飞船周围活跃 chunk 集合。
 *
 * - 飞船移动时，每 250ms 检查一次"应激活的 chunk 集合"，对比当前集合，加载新 chunk、卸载远 chunk
 * - chunk group 挂在 playerGroup 下，position = chunkAnchor (world) - playerGroup.position（Float32 安全）
 * - 不做 LOD（MVP 7 简化），所有活跃 chunk 相同细节
 */

import * as THREE from 'three/webgpu'
import { ACTIVE_CHUNK_RADIUS, CHUNK_SIZE, chunkOrigin, worldPosToChunk } from '../universe/chunk'
import { generateChunk } from '../universe/generateChunk'
import type { ChunkData } from '../universe/generateChunk'
import { buildChunk, type ChunkHandle } from './buildChunk'

interface ChunkEntry {
  handle: ChunkHandle
  position: THREE.Vector3 // 在 playerGroup 内的局部位置（= chunkWorld - playerPos）
  cx: number
  cy: number
  cz: number
  lastSeen: number // 最近一次"应激活"时的时间戳
}

function chunkKey(cx: number, cy: number, cz: number): string {
  return `${cx},${cy},${cz}`
}

export class ChunkManager {
  private playerGroup: THREE.Group
  private chunks = new Map<string, ChunkEntry>()
  private lastSyncMs = 0
  private syncIntervalMs = 250
  private orbitTime = 0

  constructor(playerGroup: THREE.Group) {
    this.playerGroup = playerGroup
  }

  /** 每帧调用 */
  update(_dt: number, time: number): void {
    this.orbitTime = time
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
      this.chunks.set(key, {
        handle,
        position: localPos,
        cx,
        cy,
        cz,
        lastSeen: performance.now(),
      })
    }

    // 更新已有 chunk 的位置（跟随 player 平移）
    for (const entry of this.chunks.values()) {
      const [ox, oy, oz] = chunkOrigin(entry.cx, entry.cy, entry.cz)
      const newLocalX = ox - px
      const newLocalY = oy - py
      const newLocalZ = oz - pz
      entry.handle.group.position.set(newLocalX, newLocalY, newLocalZ)
      entry.position.set(newLocalX, newLocalY, newLocalZ)
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

  dispose(): void {
    for (const entry of this.chunks.values()) {
      entry.handle.dispose()
    }
    this.chunks.clear()
  }
}