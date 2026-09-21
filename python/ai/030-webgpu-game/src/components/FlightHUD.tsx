/**
 * 飞行 HUD：准星 + 位置 + 速度 + chunk count + 区块编号。
 */

import { useApp } from '../state/AppState'
import { worldPosToChunk } from '../universe/chunk'

export function FlightHUD() {
  const { state } = useApp()
  const [cx, cy, cz] = worldPosToChunk(state.position.x, state.position.y, state.position.z)

  return (
    <>
      {/* 中央十字准星 */}
      <div className="reticle" aria-hidden="true">
        <span className="reticle-h" />
        <span className="reticle-v" />
      </div>

      {/* 左下：飞行数据 */}
      <div className="flight-data panel">
        <div className="flight-data-row">
          <span className="flight-data-label">位置</span>
          <code className="flight-data-value">
            ({state.position.x.toFixed(0)}, {state.position.y.toFixed(0)}, {state.position.z.toFixed(0)})
          </code>
        </div>
        <div className="flight-data-row">
          <span className="flight-data-label">区块</span>
          <code className="flight-data-value">
            ({cx}, {cy}, {cz})
          </code>
        </div>
        <div className="flight-data-row">
          <span className="flight-data-label">速度</span>
          <span className="flight-data-value">
            {state.speedMode === 'boost' ? '⚡ 加速' : '巡航'}
          </span>
        </div>
        <div className="flight-data-row">
          <span className="flight-data-label">渲染</span>
          <span className="flight-data-value">{state.chunkCount} chunks</span>
        </div>
      </div>

      {/* 右下：控制提示 */}
      <div className="controls-help panel">
        <div className="help-row">
          <kbd>W</kbd>
          <kbd>A</kbd>
          <kbd>S</kbd>
          <kbd>D</kbd>
          <span className="help-text">平移</span>
        </div>
        <div className="help-row">
          <kbd>Space</kbd>
          <kbd>Ctrl</kbd>
          <span className="help-text">上升 / 下降</span>
        </div>
        <div className="help-row">
          <kbd>Shift</kbd>
          <span className="help-text">加速 ×6</span>
        </div>
        <div className="help-row">
          <kbd>Mouse</kbd>
          <span className="help-text">视角</span>
        </div>
        <div className="help-row">
          <kbd>T</kbd>
          <span className="help-text">传送</span>
        </div>
        <div className="help-row">
          <kbd>Esc</kbd>
          <span className="help-text">暂停 / 主菜单</span>
        </div>
      </div>
    </>
  )
}