/**
 * 天体信息面板：名称、类型、关键数据、儿童科普介绍、小知识。
 * 显示内容受设置项 showRealData / showFacts 控制。
 */

import { BODIES, MOON_DISTANCE_EARTH_KM, SUN_AGE_BILLION_YEARS, type BodyData } from '../data/planetData'
import { BODY_TEXTS } from '../data/uiText'
import { formatDistanceKm, formatPeriodDays, formatRadiusKm } from '../utils/format'
import { useApp } from '../state/AppState'
import { playClose } from '../utils/audio'

const TYPE_ZH: Record<BodyData['type'], string> = {
  star: '恒星',
  rocky: '岩石行星',
  gasGiant: '气态巨行星',
  iceGiant: '冰巨星',
  moon: '卫星',
}

function rotationText(d: BodyData): string {
  const h = Math.abs(d.rotationPeriodHours)
  const retro = d.rotationPeriodHours < 0 ? '（逆向）' : ''
  if (d.id === 'sun') return '约 25 天'
  if (h >= 48) return `${(h / 24).toFixed(h >= 240 ? 0 : 1)} 天${retro}`
  return `${h.toFixed(h >= 10 ? 0 : 1)} 小时${retro}`
}

export function InfoPanel() {
  const { state, dispatch, sceneRef } = useApp()
  const id = state.selectedId
  if (!id) return null
  const data = BODIES[id]
  const text = BODY_TEXTS[id]
  const { settings } = state

  return (
    <section className="info-panel panel" aria-label={`${data.nameZh} 详情`} role="dialog" aria-modal="false">
      <div className="info-head">
        <span className="info-emoji" aria-hidden="true">
          {data.emoji}
        </span>
        <div className="info-title">
          <h2 className="info-name">
            {data.nameZh}
            <span style={{ fontSize: 13, color: 'var(--text-dim)', fontWeight: 500, marginLeft: 6 }}>
              {data.nameEn}
            </span>
          </h2>
          <div className="info-tagline">{text.tagline}</div>
        </div>
        <button
          className="info-close"
          aria-label="关闭信息面板"
          onClick={() => {
            sceneRef.current?.selectBody(null) // 同步清除场景高亮
            dispatch({ type: 'SELECT_BODY', id: null })
            playClose()
          }}
        >
          ✕
        </button>
      </div>

      <span className="type-chip">{TYPE_ZH[data.type]}</span>

      {settings.showRealData && (
        <div className="info-stats">
          <div className="stat-card">
            <div className="stat-label">直径</div>
            <div className="stat-value">{formatRadiusKm(data.radiusKm * 2)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">
              {data.id === 'moon' ? '距地球' : data.id === 'sun' ? '距地球' : '距太阳'}
            </div>
            <div className="stat-value">
              {data.id === 'moon'
                ? formatDistanceKm(MOON_DISTANCE_EARTH_KM)
                : data.id === 'sun'
                  ? formatDistanceKm(149_600_000)
                  : formatDistanceKm(data.distanceFromSunMkm! * 1e6)}
            </div>
          </div>
          {data.orbitalPeriodDays > 0 && (
            <div className="stat-card">
              <div className="stat-label">{data.id === 'moon' ? '绕地球周期' : '公转周期'}</div>
              <div className="stat-value">{formatPeriodDays(data.orbitalPeriodDays)}</div>
            </div>
          )}
          <div className="stat-card">
            <div className="stat-label">自转周期</div>
            <div className="stat-value">{rotationText(data)}</div>
          </div>
          {data.surfaceTempC !== undefined && (
            <div className="stat-card">
              <div className="stat-label">温度</div>
              <div className="stat-value">
                {data.surfaceTempC >= 1000 ? '约 5500 ℃' : `${data.surfaceTempC} ℃`}
              </div>
            </div>
          )}
          {data.id === 'sun' ? (
            <div className="stat-card">
              <div className="stat-label">年龄</div>
              <div className="stat-value">约 {SUN_AGE_BILLION_YEARS} 亿年</div>
            </div>
          ) : data.moons > 0 ? (
            <div className="stat-card">
              <div className="stat-label">卫星数</div>
              <div className="stat-value">{data.moons} 颗</div>
            </div>
          ) : (
            <div className="stat-card">
              <div className="stat-label">卫星数</div>
              <div className="stat-value">0 颗</div>
            </div>
          )}
        </div>
      )}

      <p className="info-desc">{text.description}</p>

      {settings.showFacts && text.facts.length > 0 && (
        <div className="info-facts">
          {text.facts.slice(0, 3).map((f, i) => (
            <div className="fact-row" key={i}>
              💡 {f}
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
