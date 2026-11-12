/**
 * 应用布局：3D 画布 + 全部 UI 层。
 */

import { SolarSystemCanvas } from './components/SolarSystemCanvas'
import { TopBar } from './components/TopBar'
import { PlanetNav } from './components/PlanetNav'
import { ControlBar } from './components/ControlBar'
import { InfoPanel } from './components/InfoPanel'
import { SettingsPanel } from './components/SettingsPanel'
import { FunFactCard } from './components/FunFactCard'
import { LoadingScreen } from './components/LoadingScreen'
import { WelcomeHint } from './components/WelcomeHint'
import { useApp } from './state/AppState'

export default function App() {
  const { state } = useApp()

  return (
    <div className="app-shell">
      <SolarSystemCanvas />

      {state.fatalError ? (
        <div className="error-screen" role="alert">
          <div className="error-icon" aria-hidden="true">
            🛸
          </div>
          <p className="loading-title">哎呀，飞船启动失败了</p>
          <p className="error-msg">{state.fatalError}</p>
          <button className="retry-btn" onClick={() => location.reload()}>
            重新尝试
          </button>
        </div>
      ) : (
        <>
          {!state.ready && <LoadingScreen />}
          {state.ready && <WelcomeHint />}
          {state.ready && state.degraded && (
            <div className="toast" role="status">
              ℹ️ 当前浏览器不支持 WebGPU，已自动切换到兼容模式
            </div>
          )}
          <TopBar />
          <PlanetNav />
          <InfoPanel />
          <FunFactCard />
          <ControlBar />
          <SettingsPanel />
        </>
      )}
    </div>
  )
}
