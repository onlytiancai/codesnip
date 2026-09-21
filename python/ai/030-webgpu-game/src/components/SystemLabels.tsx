/**
 * SystemLabels：DOM overlay 显示附近星系的编号 + 距离。
 *
 * 每 100ms 通过 sceneApi.getNearbyLabels() 更新位置。
 * 显示阈值：玩家距离 < MAX_RADIUS scene 单位（可调）。
 * 显示数量上限：MAX_LABELS 个最近的星系。
 *
 * 视野内恒星：全亮显示；视野外恒星：clamp 到屏幕边缘，半透明，提示方向。
 */

import { useEffect, useState } from 'react'
import { useApp, type SystemLabel } from '../state/AppState'

const MAX_LABELS = 8
const MAX_RADIUS = 1200

export function SystemLabels() {
  const { sceneApiRef } = useApp()
  const [labels, setLabels] = useState<SystemLabel[]>([])
  const [viewport, setViewport] = useState({ w: 0, h: 0 })

  useEffect(() => {
    const id = setInterval(() => {
      const list = sceneApiRef.current?.getNearbyLabels(MAX_RADIUS) ?? []
      setLabels(list.slice(0, MAX_LABELS))
      setViewport({ w: window.innerWidth, h: window.innerHeight })
    }, 100)
    return () => clearInterval(id)
  }, [sceneApiRef])

  return (
    <div className="system-labels" aria-hidden="true">
      {labels.map((l) => {
        // 视野内：完全显示；视野外：clamp 到屏幕边缘，半透明，提示方向
        const margin = 60
        const maxX = viewport.w - margin
        const maxY = viewport.h - margin
        const x = Math.max(margin, Math.min(maxX, l.screenX))
        const y = Math.max(margin, Math.min(maxY, l.screenY))
        const clamped = x !== l.screenX || y !== l.screenY
        const fade = l.inFront && !clamped ? 1 : clamped ? 0.55 : 0
        const distText = l.distance > 1 ? l.distance.toFixed(0) : l.distance.toFixed(1)
        return (
          <div
            key={l.id}
            className="system-label"
            style={{
              transform: `translate(${x}px, ${y}px)`,
              opacity: fade,
            }}
          >
            <div className="system-label-id">#{l.id.toString(16).toUpperCase().padStart(4, '0')}</div>
            <div className="system-label-dist">{distText} u</div>
          </div>
        )
      })}
    </div>
  )
}