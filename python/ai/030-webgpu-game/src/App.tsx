/**
 * 应用布局：MVP 7+ 自由飞行版。
 */

import { LoadingScreen } from './components/LoadingScreen'
import { FlightHUD } from './components/FlightHUD'
import { StartOverlay } from './components/StartOverlay'
import { TeleportDialog } from './components/TeleportDialog'
import { HistoryPanel } from './components/HistoryPanel'
import { SystemLabels } from './components/SystemLabels'
import { UniverseCanvas } from './components/UniverseCanvas'
import { useApp } from './state/AppState'

export default function App() {
  const { state } = useApp()

  return (
    <div className="app-shell">
      {state.fatalError ? (
        <div className="error-screen" role="alert">
          <div className="error-icon" aria-hidden="true">🛸</div>
          <p className="loading-title">哎呀，超空间引擎启动失败</p>
          <p className="error-msg">{state.fatalError}</p>
          <button className="retry-btn" onClick={() => location.reload()}>重新尝试</button>
        </div>
      ) : !state.ready ? (
        <>
          <LoadingScreen />
          <UniverseCanvas />
        </>
      ) : (
        <>
          <UniverseCanvas />
          <SystemLabels />
          <FlightHUD />
          <HistoryPanel />
          <TeleportDialog />
          <StartOverlay />
        </>
      )}
    </div>
  )
}