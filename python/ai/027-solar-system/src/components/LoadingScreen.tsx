/**
 * 启动 Loading 屏：阶段化提示文字 + 火箭动画。
 */

import { useEffect, useState } from 'react'

const STEPS = [
  '🚀 正在启动太阳系...',
  '🌍 正在生成行星纹理...',
  '🪐 正在布置行星轨道...',
  '✨ 正在点亮星空...',
]

export function LoadingScreen() {
  const [step, setStep] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => setStep((s) => Math.min(s + 1, STEPS.length - 1)), 900)
    return () => clearInterval(timer)
  }, [])

  return (
    <div className="loading-screen" role="status" aria-live="polite">
      <div className="loading-rocket" aria-hidden="true">
        🚀
      </div>
      <p className="loading-title">太阳系探索</p>
      <div className="loading-steps">
        {STEPS[step]}
        <span className="loading-dots" aria-hidden="true" />
      </div>
    </div>
  )
}
