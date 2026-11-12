/**
 * 3D 场景挂载点：创建 SceneManager，桥接 React 状态与场景指令。
 */

import { useEffect, useRef } from 'react'
import { SceneManager } from '../scene/SceneManager'
import { useApp } from '../state/AppState'

export function SolarSystemCanvas() {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { dispatch, sceneRef, state } = useApp()

  useEffect(() => {
    const container = containerRef.current!
    const canvas = canvasRef.current!
    const manager = new SceneManager(container, canvas, {
      onReady: (backend, degraded) => dispatch({ type: 'SET_BACKEND', backend, degraded }),
      onSelect: (id) => dispatch({ type: 'SELECT_BODY', id }),
      onQualitySlow: () => dispatch({ type: 'SET_RESOLVED_QUALITY', quality: 'low' }),
      onFatal: (message) => dispatch({ type: 'SET_FATAL', message }),
    })
    sceneRef.current = manager
    // 用最新设置初始化（含 localStorage 恢复的设置）
    void manager.init(state.settings).then(() => {
      dispatch({ type: 'SET_READY' })
    })
    return () => {
      sceneRef.current = null
      manager.dispose()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 初始设置需要在 manager 创建后应用一次（init 内部已做，这里补 resolvedQuality 变化）
  useEffect(() => {
    sceneRef.current?.setResolvedQuality(state.resolvedQuality)
  }, [state.resolvedQuality, sceneRef])

  return (
    <div ref={containerRef} className="scene-container" aria-hidden="true">
      <canvas ref={canvasRef} />
    </div>
  )
}
