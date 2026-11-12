/**
 * IndexedDB 封装。
 *
 * 数据库：`webgpu-universe`，版本 2
 *   - history: 重生点（玩家手动保存的位置），{ id(auto), position, quaternion, label, savedAt }
 *   - playerState: 当前飞船状态（位置 + 视角），key='last'
 */

const DB_NAME = 'webgpu-universe'
const DB_VERSION = 2
const STORE_HISTORY = 'history'
const STORE_PLAYER = 'playerState'

export interface Position3 {
  x: number
  y: number
  z: number
}

export interface Quaternion4 {
  x: number
  y: number
  z: number
  w: number
}

export interface HistoryEntry {
  id?: number
  position: Position3
  quaternion: Quaternion4
  label: string
  savedAt: number
}

let dbPromise: Promise<IDBDatabase> | null = null

export function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE_HISTORY)) {
        const s = db.createObjectStore(STORE_HISTORY, { keyPath: 'id', autoIncrement: true })
        s.createIndex('by-time', 'savedAt')
      }
      if (!db.objectStoreNames.contains(STORE_PLAYER)) {
        db.createObjectStore(STORE_PLAYER)
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
  return dbPromise
}

/** 追加一条 history。返回新 id。 */
export async function addHistory(entry: Omit<HistoryEntry, 'id'>): Promise<number | null> {
  try {
    const db = await openDb()
    return await new Promise((resolve) => {
      const tx = db.transaction(STORE_HISTORY, 'readwrite')
      const req = tx.objectStore(STORE_HISTORY).add(entry)
      req.onsuccess = () => resolve(typeof req.result === 'number' ? req.result : null)
      req.onerror = () => resolve(null)
    })
  } catch {
    return null
  }
}

/** 取所有 history（按 savedAt 倒序） */
export async function getRecentHistory(limit = 50): Promise<HistoryEntry[]> {
  try {
    const db = await openDb()
    return await new Promise((resolve) => {
      const tx = db.transaction(STORE_HISTORY, 'readonly')
      const idx = tx.objectStore(STORE_HISTORY).index('by-time')
      const out: HistoryEntry[] = []
      const req = idx.openCursor(null, 'prev')
      req.onsuccess = () => {
        const cur = req.result
        if (!cur || out.length >= limit) {
          resolve(out)
          return
        }
        out.push(cur.value as HistoryEntry)
        cur.continue()
      }
      req.onerror = () => resolve([])
    })
  } catch {
    return []
  }
}

/** 删除一条 history */
export async function deleteHistory(id: number): Promise<void> {
  try {
    const db = await openDb()
    await new Promise<void>((resolve) => {
      const tx = db.transaction(STORE_HISTORY, 'readwrite')
      tx.objectStore(STORE_HISTORY).delete(id)
      tx.oncomplete = () => resolve()
      tx.onerror = () => resolve()
    })
  } catch {
    /* ignore */
  }
}

/** 导出整库为 JSON 对象 */
export async function exportAll(): Promise<{
  exportedAt: number
  version: number
  history: HistoryEntry[]
}> {
  try {
    const db = await openDb()
    const history = await new Promise<HistoryEntry[]>((resolve) => {
      const tx = db.transaction(STORE_HISTORY, 'readonly')
      const req = tx.objectStore(STORE_HISTORY).getAll()
      req.onsuccess = () => resolve(req.result as HistoryEntry[])
      req.onerror = () => resolve([])
    })
    return {
      exportedAt: Date.now(),
      version: DB_VERSION,
      history,
    }
  } catch {
    return { exportedAt: Date.now(), version: DB_VERSION, history: [] }
  }
}

/** 清空并导入（先清后写） */
export async function importAll(payload: { history?: HistoryEntry[] }): Promise<number> {
  try {
    const db = await openDb()
    await new Promise<void>((resolve) => {
      const tx = db.transaction(STORE_HISTORY, 'readwrite')
      tx.objectStore(STORE_HISTORY).clear()
      tx.oncomplete = () => resolve()
      tx.onerror = () => resolve()
    })
    let count = 0
    if (payload.history && payload.history.length) {
      const db2 = await openDb()
      await new Promise<void>((resolve) => {
        const tx = db2.transaction(STORE_HISTORY, 'readwrite')
        const store = tx.objectStore(STORE_HISTORY)
        for (const e of payload.history!) {
          const { id: _omit, ...rest } = e
          store.add(rest)
          count++
        }
        tx.oncomplete = () => resolve()
        tx.onerror = () => resolve()
      })
    }
    return count
  } catch {
    return 0
  }
}