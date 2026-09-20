/**
 * 启动 Loading 屏：阶段化提示文字 + 火箭动画。
 */

import { useEffect, useState } from 'react'

const STEPS = [
  '🚀 启动超空间引擎...',
  '🪐 正在计算坐标种子...',
  '🌟 正在生成恒星与行星...',
  '✨ 正在点亮整片星空...',
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
        🌌
      </div>
      <p className="loading-title">序列化星空</p>
      <div className="loading-steps">
        {STEPS[step]}
        <span className="loading-dots" aria-hidden="true" />
      </div>
    </div>
  )
}
