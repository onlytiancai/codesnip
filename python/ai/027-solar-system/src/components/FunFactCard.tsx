/**
 * 💡 你知道吗：浮动小知识卡片，定时轮换，可关闭。
 */

import { useEffect, useState } from 'react'
import { FUN_FACTS } from '../data/uiText'
import { useApp } from '../state/AppState'

export function FunFactCard() {
  const { state } = useApp()
  const [index, setIndex] = useState(0)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    const timer = setInterval(() => setIndex((i) => (i + 1) % FUN_FACTS.length), 9000)
    return () => clearInterval(timer)
  }, [])

  if (dismissed || !state.settings.showFacts || state.selectedId) return null

  return (
    <div className="funfact" role="note" aria-label="小知识">
      <button className="funfact-dismiss" aria-label="关闭小知识" onClick={() => setDismissed(true)}>
        ✕
      </button>
      {FUN_FACTS[index]}
    </div>
  )
}
