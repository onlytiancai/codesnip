/**
 * 左侧行星导航：点击直接飞过去。移动端收纳为抽屉。
 */

import { useState } from 'react'
import { BODIES, NAV_ORDER } from '../data/planetData'
import { useApp } from '../state/AppState'
import { playClick } from '../utils/audio'

export function PlanetNav() {
  const { state, flyTo } = useApp()
  const [drawerOpen, setDrawerOpen] = useState(false)

  return (
    <>
      <aside className={`nav-panel panel${drawerOpen ? ' nav-drawer-open' : ''}`} aria-label="天体导航">
        {NAV_ORDER.map((id) => {
          const d = BODIES[id]
          return (
            <button
              key={id}
              className={`nav-btn${state.selectedId === id ? ' active' : ''}`}
              aria-label={`飞向${d.nameZh}`}
              aria-current={state.selectedId === id}
              onClick={() => {
                flyTo(id)
                setDrawerOpen(false)
                playClick()
              }}
            >
              <span className="nav-emoji" aria-hidden="true">
                {d.emoji}
              </span>
              <span className="nav-dot" style={{ background: d.accent, color: d.accent }} aria-hidden="true" />
              <span>{d.nameZh}</span>
            </button>
          )
        })}
      </aside>
      <button
        className="icon-btn nav-hamburger"
        aria-label={drawerOpen ? '收起天体导航' : '展开天体导航'}
        onClick={() => {
          setDrawerOpen((v) => !v)
          playClick()
        }}
      >
        {drawerOpen ? '✕' : '🪐'}
      </button>
      {drawerOpen && <div className="nav-backdrop" onClick={() => setDrawerOpen(false)} aria-hidden="true" />}
    </>
  )
}
