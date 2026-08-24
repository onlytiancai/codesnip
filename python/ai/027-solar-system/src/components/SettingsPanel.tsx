/**
 * 设置抽屉：显示 / 动画 / 视觉 / 科普 / 音效。
 */

import { useApp, type AppSettings, type QualitySetting } from '../state/AppState'
import { playClick, playClose, setSoundEnabled } from '../utils/audio'

interface ToggleRowProps {
  label: string
  hint?: string
  checked: boolean
  onChange: (v: boolean) => void
}

function ToggleRow({ label, hint, checked, onChange }: ToggleRowProps) {
  return (
    <label className="setting-row">
      <span className="setting-label">
        {label}
        {hint && <span className="setting-hint">{hint}</span>}
      </span>
      <input type="checkbox" className="switch-input" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="switch" aria-hidden="true" />
    </label>
  )
}

const QUALITY_OPTIONS: { key: QualitySetting; label: string }[] = [
  { key: 'auto', label: '自动' },
  { key: 'high', label: '高质量' },
  { key: 'low', label: '性能' },
]

export function SettingsPanel() {
  const { state, dispatch } = useApp()
  const s = state.settings

  if (!state.settingsOpen) return null

  const set = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    dispatch({ type: 'SET_SETTING', key, value: value as never })
  }

  return (
    <>
      <div
        className="settings-backdrop"
        aria-hidden="true"
        onClick={() => {
          dispatch({ type: 'SET_SETTINGS_OPEN', open: false })
          playCloseSafe()
        }}
      />
      <aside className="settings-panel" role="dialog" aria-label="设置">
        <div className="settings-head">
          <h2 className="settings-title">⚙️ 设置</h2>
          <button
            className="icon-btn"
            style={{ width: 34, height: 34, fontSize: 15 }}
            aria-label="关闭设置"
            onClick={() => {
              dispatch({ type: 'SET_SETTINGS_OPEN', open: false })
              playCloseSafe()
            }}
          >
            ✕
          </button>
        </div>

        <div className="settings-group">
          <h3 className="settings-group-title">显示</h3>
          <ToggleRow label="轨道线" hint="显示行星运行轨道" checked={s.showOrbits} onChange={(v) => set('showOrbits', v)} />
          <ToggleRow label="行星名称" hint="3D 场景中的名称标签" checked={s.showLabels} onChange={(v) => set('showLabels', v)} />
          <ToggleRow label="名称下方数据" hint="距离 / 周期小字" checked={s.showData} onChange={(v) => set('showData', v)} />
          <ToggleRow label="星空背景" checked={s.showStarfield} onChange={(v) => set('showStarfield', v)} />
          <ToggleRow label="参考网格" hint="轨道平面辅助线" checked={s.showGuide} onChange={(v) => set('showGuide', v)} />
        </div>

        <div className="settings-group">
          <h3 className="settings-group-title">动画</h3>
          <ToggleRow label="自动旋转视角" checked={s.autoRotate} onChange={(v) => set('autoRotate', v)} />
          <ToggleRow label="行星公转" checked={s.orbitEnabled} onChange={(v) => set('orbitEnabled', v)} />
          <ToggleRow label="行星自转" checked={s.spinEnabled} onChange={(v) => set('spinEnabled', v)} />
        </div>

        <div className="settings-group">
          <h3 className="settings-group-title">视觉</h3>
          <ToggleRow label="辉光效果" hint="太阳光晕增强（bloom）" checked={s.bloomEnabled} onChange={(v) => set('bloomEnabled', v)} />
          <ToggleRow label="阴影" hint="性能开销较大，建议关闭" checked={s.shadowsEnabled} onChange={(v) => set('shadowsEnabled', v)} />
          <div className="setting-row">
            <span className="setting-label">
              质量模式
              <span className="setting-hint">低性能设备建议选「性能」</span>
            </span>
            <div className="segmented" role="radiogroup" aria-label="质量模式">
              {QUALITY_OPTIONS.map((o) => (
                <button
                  key={o.key}
                  className={s.quality === o.key ? 'active' : ''}
                  role="radio"
                  aria-checked={s.quality === o.key}
                  onClick={() => {
                    set('quality', o.key)
                    playClick()
                  }}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="settings-group">
          <h3 className="settings-group-title">科普</h3>
          <ToggleRow label="真实数据" hint="信息面板中的直径、距离等" checked={s.showRealData} onChange={(v) => set('showRealData', v)} />
          <ToggleRow label="趣味小知识" checked={s.showFacts} onChange={(v) => set('showFacts', v)} />
          <ToggleRow label="标签显示距离" checked={s.showDistances} onChange={(v) => set('showDistances', v)} />
          <ToggleRow label="标签显示周期" checked={s.showPeriods} onChange={(v) => set('showPeriods', v)} />
        </div>

        <div className="settings-group">
          <h3 className="settings-group-title">音效</h3>
          <ToggleRow
            label="交互音效"
            hint="默认关闭，开启后播放轻微点击音"
            checked={s.soundEnabled}
            onChange={(v) => {
              set('soundEnabled', v)
              setSoundEnabled(v)
              if (v) playClick()
            }}
          />
        </div>
      </aside>
    </>
  )
}

function playCloseSafe(): void {
  playClose()
}
