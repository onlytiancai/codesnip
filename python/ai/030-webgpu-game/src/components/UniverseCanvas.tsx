/**
 * 3D 画布挂载：创建 canvas、初始化 SceneManager、桥接 player state → AppState。
 *
 * 点击 canvas → 触发 pointer lock（让 mouse 控制视角）。
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
        // 只有 position 变化时才 schedule 保存（避免每 200ms 重置 5s debounce）
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
    })
    mgrRef.current = mgr

    // 暴露 SceneApi
    sceneApiRef.current = {
      teleport: (x, y, z) => {
        mgr.teleport(x, y, z)
        // teleport 后立即持久化一次
        const q = new THREE.Quaternion()
        mgr.flight.getQuaternion(q)
        scheduleSavePlayer({
          position: { x, y, z },
          quaternion: { x: q.x, y: q.y, z: q.z, w: q.w },
          updatedAt: Date.now(),
        })
      },
      requestPointerLock: () => mgr.flight.requestLock(),
    }

    void mgr.init()

    // 监听 pointer lock 状态
    const onLockChange = () => {
      setState((s) => ({ ...s, pointerLocked: document.pointerLockElement === canvas }))
    }
    document.addEventListener('pointerlockchange', onLockChange)

    return () => {
      mgr.dispose()
      mgrRef.current = null
      // 清理时把状态写入 IndexedDB
      void flushPlayer()
      document.removeEventListener('pointerlockchange', onLockChange)
      el.removeChild(canvas)
    }
  }, [setState, sceneApiRef])

  return <div ref={containerRef} className="scene-container" aria-label="3D 宇宙场景" />
}