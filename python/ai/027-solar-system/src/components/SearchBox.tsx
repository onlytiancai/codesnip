/**
 * 行星搜索：输入中文名/英文名，回车或点击结果直接飞过去。
 */

import { useMemo, useRef, useState } from 'react'
import { BODIES, NAV_ORDER, type BodyId } from '../data/planetData'
import { useApp } from '../state/AppState'
import { playClick } from '../utils/audio'

export function SearchBox() {
  const { flyTo, state } = useApp()
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const results = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return []
    return NAV_ORDER.filter((id) => {
      const d = BODIES[id]
      return d.nameZh.includes(query.trim()) || d.nameEn.toLowerCase().includes(q)
    })
  }, [query])

  const pick = (id: BodyId) => {
    flyTo(id)
    setQuery('')
    setOpen(false)
    inputRef.current?.blur()
    playClick()
  }

  return (
    <div className="search-wrap">
      <input
        ref={inputRef}
        className="search-input"
        type="search"
        role="combobox"
        aria-expanded={open}
        aria-label="搜索行星"
        placeholder="🔍 搜索行星"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value)
          setOpen(true)
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && results.length > 0) pick(results[0])
          if (e.key === 'Escape') {
            setOpen(false)
            inputRef.current?.blur()
          }
        }}
      />
      {open && query.trim() && (
        <div className="search-results" role="listbox" aria-label="搜索结果">
          {results.length === 0 ? (
            <div className="search-empty">没有找到「{query}」</div>
          ) : (
            results.map((id) => (
              <button
                key={id}
                className={`search-result${state.selectedId === id ? ' active' : ''}`}
                role="option"
                aria-selected={state.selectedId === id}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(id)}
              >
                <span>{BODIES[id].emoji}</span>
                <span>{BODIES[id].nameZh}</span>
                <span style={{ color: 'var(--text-dim)', fontSize: 11 }}>{BODIES[id].nameEn}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  )
}
