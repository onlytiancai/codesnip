/**
 * 3D 画布挂载：创建 canvas、初始化 SceneManager、桥接 player state → AppState。
 *
 * Esc 状态机由 SceneManager 内部维护并通过 onEscapeState 通知 React。
 */

import { useEffect, useRef } from 'react'
import * as THREE from 'three/webgpu'
import { SceneManager } from '../scene/SceneManager'
import { useApp } from '../state/AppState'
import { flushPlayer, scheduleSavePlayer } from '../storage/playerState'

export function UniverseCanvas() {
  const { setState, sceneApiRef } = useApp()
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mgrRef = useRef<SceneManager | null>(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const canvas = document.createElement('canvas')
    el.appendChild(canvas)

    let lastSavedKey = ''

    const mgr = new SceneManager(el, canvas, {
      onReady: (backend, degraded) =>
        setState((s) => ({ ...s, ready: true, backend, degraded })),
      onFatal: (message) => setState((s) => ({ ...s, fatalError: message })),
      onPlayerState: ({ position, speedMode }) => {
        setState((s) => ({ ...s, position: { x: position.x, y: position.y, z: position.z }, speedMode }))
        const key = `${position.x.toFixed(2)},${position.y.toFixed(2)},${position.z.toFixed(2)}`
        if (key !== lastSavedKey) {
          lastSavedKey = key
          const q = new THREE.Quaternion()
          mgr.flight.getQuaternion(q)
          scheduleSavePlayer({
            position: { x: position.x, y: position.y, z: position.z },
            quaternion: { x: q.x, y: q.y, z: q.z, w: q.w },
            updatedAt: Date.now(),
          })
        }
      },
      onChunkCount: (count) => setState((s) => ({ ...s, chunkCount: count })),
      onEscapeState: (escapeState) => setState((s) => ({ ...s, escapeState })),
    })
    mgrRef.current = mgr

    // 暴露 SceneApi
    sceneApiRef.current = {
      teleport: (x, y, z) => {
        mgr.teleport(x, y, z)
        const q = new THREE.Quaternion()
        mgr.flight.getQuaternion(q)
        scheduleSavePlayer({
          position: { x, y, z },
          quaternion: { x: q.x, y: q.y, z: q.z, w: q.w },
          updatedAt: Date.now(),
        })
      },
      // 启动或 resume 时调用：从 menu/paused 状态重新 lock pointer
      requestPointerLock: () => mgr.requestPointerLock(),
      getNearbyLabels: (maxRadius: number) => {
        const list = mgr.getNearbySystems(maxRadius)
        return list.map((s) => {
          const proj = mgr.projectToScreen(s.worldPos)
          return {
            id: s.id,
            distance: s.distance,
            screenX: proj.x,
            screenY: proj.y,
            inFront: proj.inFront,
          }
        })
      },
    }

    void mgr.init()

    return () => {
      mgr.dispose()
      mgrRef.current = null
      void flushPlayer()
      el.removeChild(canvas)
    }
  }, [setState, sceneApiRef])

  // 点击星空 → 请求 pointer lock（必须在 user gesture 内调用）
  // 在 escapeState === 'playing' 时点击是无效的（浏览器忽略），其他状态都能进入飞行
  const onCanvasClick = () => {
    sceneApiRef.current?.requestPointerLock()
  }

  return (
    <div
      ref={containerRef}
      className="scene-container"
      aria-label="3D 宇宙场景"
      onClick={onCanvasClick}
    />
  )
}