/**
 * 启动 overlay：未进入 pointer lock 时显示「点击进入飞行」提示。
 * 同时是品牌 / WebGPU 徽章 / 设置按钮的容器。
 */

import { useApp } from '../state/AppState'

export function StartOverlay() {
  const { state, sceneApiRef } = useApp()
  if (!state.ready) return null

  const onStart = () => {
    sceneApiRef.current?.requestPointerLock()
  }

  return (
    <div
      className={`start-overlay ${state.pointerLocked ? 'hidden' : ''}`}
      onClick={onStart}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onStart()
      }}
    >
      <div className="start-card panel">
        <div className="start-title">🌌 序列化星空</div>
        <div className="start-subtitle">点击进入超空间飞行</div>
        <div className="start-hints">
          <div>
            <kbd>WASD</kbd> 平移 · <kbd>Space</kbd> / <kbd>Ctrl</kbd> 上升下降
          </div>
          <div>
            <kbd>Shift</kbd> 加速 · <kbd>Mouse</kbd> 视角 · <kbd>T</kbd> 传送
          </div>
          <div>
            <kbd>Esc</kbd> 释放鼠标
          </div>
        </div>
        <div className="start-badge">
          {state.backend === 'webgpu' ? '⚡ WebGPU' : state.backend === 'webgl2' ? '🔄 WebGL 兼容' : '…'}
        </div>
      </div>
    </div>
  )
}