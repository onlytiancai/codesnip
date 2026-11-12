/**
 * 传送对话框：按 T 键打开，输入目标坐标 (x, y, z) 后传送。
 */

import { useEffect, useState } from 'react'
import { useApp } from '../state/AppState'

export function TeleportDialog() {
  const { state, sceneApiRef } = useApp()
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')
  const [err, setErr] = useState('')

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'KeyT' && state.escapeState === 'playing') {
        e.preventDefault()
        setOpen((v) => !v)
      } else if (e.code === 'Escape' && open) {
        setOpen(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [state.escapeState, open])

  const submit = () => {
    const m = text.trim().match(/^(-?\d+(\.\d+)?)[\s,]+(-?\d+(\.\d+)?)[\s,]+(-?\d+(\.\d+)?)$/)
    if (!m) {
      setErr('格式：x y z 或 x, y, z （十进制）')
      return
    }
    const x = parseFloat(m[1])
    const y = parseFloat(m[3])
    const z = parseFloat(m[5])
    sceneApiRef.current?.teleport(x, y, z)
    setOpen(false)
    setText('')
    setErr('')
  }

  if (!open) return null

  return (
    <div className="teleport-dialog panel" onClick={(e) => e.stopPropagation()}>
      <div className="coord-title">传送</div>
      <input
        type="text"
        autoFocus
        value={text}
        onChange={(e) => {
          setText(e.target.value)
          if (err) setErr('')
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') submit()
        }}
        placeholder="x y z（例：1200 0 -500）"
        className="coord-input"
        spellCheck={false}
      />
      <div className="coord-row" style={{ justifyContent: 'flex-end', marginTop: 8 }}>
        <button
          type="button"
          className="small-btn"
          onClick={() => {
            setOpen(false)
            setErr('')
          }}
        >
          取消
        </button>
        <button type="button" className="small-btn" onClick={submit}>
          GO
        </button>
      </div>
      {err && <div className="coord-err">{err}</div>}
      <div className="coord-hint">
        当前 ({state.position.x.toFixed(0)}, {state.position.y.toFixed(0)}, {state.position.z.toFixed(0)})
      </div>
    </div>
  )
}