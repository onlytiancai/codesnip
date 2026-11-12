/**
 * Player state 持久化：飞船世界坐标 + 视角四元数 + 速度模式。
 *
 * 用 IndexedDB 单条 'last' record 存储；每 5s debounce + pagehide flush。
 */

import { openDb } from './db'

export interface PlayerState {
  position: { x: number; y: number; z: number }
  quaternion: { x: number; y: number; z: number; w: number }
  updatedAt: number
}

const STORE = 'playerState'
const KEY = 'last'
let pending: PlayerState | null = null
let flushTimer: ReturnType<typeof setTimeout> | null = null

export async function loadPlayerState(): Promise<PlayerState | null> {
  try {
    const db = await openDb()
    return await new Promise((resolve) => {
      const tx = db.transaction(STORE, 'readonly')
      const req = tx.objectStore(STORE).get(KEY)
      req.onsuccess = () => resolve((req.result as PlayerState | undefined) ?? null)
      req.onerror = () => resolve(null)
    })
  } catch {
    return null
  }
}

export function scheduleSavePlayer(state: PlayerState, intervalMs = 5000) {
  pending = state
  if (flushTimer) clearTimeout(flushTimer)
  flushTimer = setTimeout(() => {
    void flushPlayer()
  }, intervalMs)
}

export async function flushPlayer(): Promise<void> {
  if (!pending) return
  const data = pending
  pending = null
  if (flushTimer) {
    clearTimeout(flushTimer)
    flushTimer = null
  }
  try {
    const db = await openDb()
    await new Promise<void>((resolve) => {
      const tx = db.transaction(STORE, 'readwrite')
      tx.objectStore(STORE).put(data, KEY)
      tx.oncomplete = () => resolve()
      tx.onerror = () => resolve()
    })
  } catch {
    /* ignore */
  }
}

// 注册 pagehide → flush
if (typeof window !== 'undefined') {
  window.addEventListener('pagehide', () => {
    void flushPlayer()
  })
}