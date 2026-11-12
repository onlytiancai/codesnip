/**
 * 底部控制栏：播放/暂停、时间速度（对数滑杆 + 预设）、重置视角。
 */

import { useApp } from '../state/AppState'
import { formatTimeScale } from '../utils/format'
import { playClick } from '../utils/audio'

// 对数映射：0.1x ~ 1000x → slider 0 ~ 100
function scaleToSlider(v: number): number {
  return ((Math.log10(v) + 1) / 4) * 100
}
function sliderToScale(x: number): number {
  return Math.pow(10, -1 + (x / 100) * 4)
}

const PRESET_SHOW = [0.1, 1, 10, 100, 1000]

export function ControlBar() {
  const { state, dispatch, sceneRef } = useApp()
  const { paused, timeScale } = state

  const setScale = (v: number) => {
    const clamped = Math.min(1000, Math.max(0.1, v))
    dispatch({ type: 'SET_TIME_SCALE', value: clamped })
  }

  return (
    <div className="controlbar panel" role="toolbar" aria-label="动画控制">
      <button
        className="play-btn"
        aria-label={paused ? '播放' : '暂停'}
        title={paused ? '播放' : '暂停'}
        onClick={() => {
          dispatch({ type: 'TOGGLE_PAUSED' })
          playClick()
        }}
      >
        {paused ? '▶️' : '⏸️'}
      </button>

      <div className="speed-control">
        <span className="speed-label" aria-live="polite">
          {formatTimeScale(timeScale)}
        </span>
        <input
          className="speed-slider"
          type="range"
          min={0}
          max={100}
          step={0.1}
          value={scaleToSlider(timeScale)}
          aria-label="时间速度"
          onChange={(e) => setScale(sliderToScale(Number(e.target.value)))}
        />
        <div className="speed-presets">
          {PRESET_SHOW.map((p) => (
            <button
              key={p}
              className={`preset-btn${Math.abs(timeScale - p) < 0.001 ? ' active' : ''}`}
              aria-label={`速度 ${formatTimeScale(p)}`}
              aria-pressed={Math.abs(timeScale - p) < 0.001}
              onClick={() => {
                setScale(p)
                playClick()
              }}
            >
              {formatTimeScale(p)}
            </button>
          ))}
        </div>
      </div>

      <div className="control-divider" aria-hidden="true" />

      <button
        className="preset-btn"
        style={{ fontSize: 13, padding: '7px 12px' }}
        onClick={() => {
          sceneRef.current?.resetView()
          playClick()
        }}
        aria-label="重置视角"
        title="回到太阳系全景"
      >
        🔭 全景
      </button>
    </div>
  )
}
