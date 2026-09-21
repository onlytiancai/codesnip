/**
 * PauseOverlay：Esc 暂停时显示「序列化星空」+「按 Esc 关闭 / 点击继续飞行」。
 *
 * - escapeState='paused'  → 显示
 * - escapeState='menu'    → 隐藏（显示主体内容，鼠标未锁）
 * - escapeState='playing' → 隐藏
 */

import { useApp } from '../state/AppState'

export function StartOverlay() {
  const { state, sceneApiRef } = useApp()
  if (!state.ready) return null
  if (state.escapeState !== 'paused') return null

  const onResume = () => {
    sceneApiRef.current?.requestPointerLock()
  }

  return (
    <div
      className="pause-overlay"
      onClick={onResume}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onResume()
      }}
    >
      <div className="pause-card panel">
        <div className="pause-title">🌌 序列化星空</div>
        <div className="pause-subtitle">飞行已暂停</div>
        <div className="pause-hints">
          <div>
            <kbd>Esc</kbd> 关闭提示框 · <kbd>Click</kbd> 继续飞行
          </div>
        </div>
        <div className="pause-badge">
          {state.backend === 'webgpu' ? '⚡ WebGPU' : state.backend === 'webgl2' ? '🔄 WebGL 兼容' : '…'}
        </div>
      </div>
    </div>
  )
}