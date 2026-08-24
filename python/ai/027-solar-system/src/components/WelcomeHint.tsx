/**
 * 欢迎提示：几秒后自动淡出；任何点击提前消失。
 */

import { useEffect } from 'react'
import { useApp } from '../state/AppState'

export function WelcomeHint() {
  const { state, dispatch } = useApp()

  useEffect(() => {
    if (state.welcomeDone) return
    const dismiss = () => dispatch({ type: 'DISMISS_WELCOME' })
    const timer = setTimeout(dismiss, 4800)
    window.addEventListener('pointerdown', dismiss, { once: true })
    return () => {
      clearTimeout(timer)
      window.removeEventListener('pointerdown', dismiss)
    }
  }, [state.welcomeDone, dispatch])

  if (state.welcomeDone) return null

  return (
    <div className="welcome" role="status" aria-label="欢迎提示">
      <p className="welcome-title">🌌 欢迎来到太阳系！</p>
      <p className="welcome-sub">拖动鼠标旋转视角 · 滚轮缩放 · 点击行星看介绍 · 双击聚焦</p>
    </div>
  )
}
