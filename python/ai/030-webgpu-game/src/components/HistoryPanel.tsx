/**
 * 重生点面板：玩家手动保存的位置。
 * MVP 9 简化版：read-only（不提供"保存当前位置为新重生点"按钮，由 T 键传送时自动添加）。
 */

import { useEffect, useState } from 'react'
import { useApp } from '../state/AppState'
import { getRecentHistory, type HistoryEntry } from '../storage/db'

export function HistoryPanel() {
  const { sceneApiRef, state } = useApp()
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [open, setOpen] = useState(false)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    let cancelled = false
    void getRecentHistory(20).then((list) => {
      if (!cancelled) setEntries(list)
    })
    return () => {
      cancelled = true
    }
  }, [tick, state.position.x, state.position.y, state.position.z])

  return (
    <div className={`history-panel ${open ? 'open' : ''}`}>
      <button type="button" className="history-toggle" onClick={() => setOpen((v) => !v)}>
        📍{' '}
        <span>
          {entries.length > 0 ? `重生点 (${entries.length})` : '重生点'}
        </span>
      </button>
      {open && (
        <div className="history-list panel">
          {entries.length === 0 ? (
            <div className="history-empty">按 T 传送时自动记录</div>
          ) : (
            entries.map((e) => (
              <button
                key={e.id ?? `${e.position.x}-${e.position.y}-${e.position.z}-${e.savedAt}`}
                type="button"
                className="history-item"
                onClick={() => {
                  sceneApiRef.current?.teleport(e.position.x, e.position.y, e.position.z)
                  setOpen(false)
                  setTick((t) => t + 1)
                }}
              >
                <code className="history-coord">
                  ({e.position.x.toFixed(0)}, {e.position.y.toFixed(0)}, {e.position.z.toFixed(0)})
                </code>
                <span className="history-name">{e.label}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  )
}