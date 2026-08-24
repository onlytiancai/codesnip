/**
 * 顶栏：品牌 + 渲染器徽章 + 显示模式切换 + 搜索 + 设置入口。
 */

import { useApp, type ViewMode } from '../state/AppState'
import { SearchBox } from './SearchBox'
import { playClick } from '../utils/audio'

const MODES: { key: ViewMode; label: string }[] = [
  { key: 'explore', label: '🧭 探索' },
  { key: 'science', label: '🔬 科普' },
  { key: 'orbits', label: '💫 轨道' },
]

export function TopBar() {
  const { state, dispatch } = useApp()
  return (
    <header className="topbar">
      <div className="brand panel">
        <span aria-hidden="true">☀️</span>
        <span>太阳系探索</span>
        {state.backend && (
          <span
            className={`badge${state.backend === 'webgl2' ? ' webgl' : ''}`}
            title={
              state.backend === 'webgpu'
                ? '正在使用 WebGPU 渲染'
                : 'WebGPU 不可用，已使用 WebGL 兼容模式'
            }
          >
            {state.backend === 'webgpu' ? '⚡ WebGPU' : '兼容模式'}
          </span>
        )}
      </div>

      <nav className="topbar-center panel" aria-label="显示模式">
        {MODES.map((m) => (
          <button
            key={m.key}
            className={`mode-btn${state.viewMode === m.key ? ' active' : ''}`}
            aria-pressed={state.viewMode === m.key}
            onClick={() => {
              dispatch({ type: 'SET_VIEW_MODE', mode: m.key })
              playClick()
            }}
          >
            {m.label}
          </button>
        ))}
      </nav>

      <div className="topbar-right">
        <SearchBox />
        <button
          className="icon-btn"
          aria-label="打开设置"
          title="设置"
          onClick={() => {
            dispatch({ type: 'SET_SETTINGS_OPEN', open: true })
            playClick()
          }}
        >
          ⚙️
        </button>
      </div>
    </header>
  )
}
