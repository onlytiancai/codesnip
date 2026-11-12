/**
 * 入口：Provider + App。
 *
 * 注意：不使用 StrictMode —— 其双挂载行为会与 WebGPU 渲染器的异步 init/dispose 冲突。
 * ready 切换由 SceneManager.onReady 触发。
 */

import { createRoot } from 'react-dom/client'
import { AppStateProvider } from './state/AppState'
import App from './App'
import './styles/global.css'

createRoot(document.getElementById('root')!).render(
  <AppStateProvider>
    <App />
  </AppStateProvider>,
)