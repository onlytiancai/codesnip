import { createRoot } from 'react-dom/client'
import { AppStateProvider } from './state/AppState'
import App from './App'
import './styles/global.css'

// 注意：不使用 StrictMode —— 其双挂载行为会与 WebGPU 渲染器的异步 init/dispose 冲突
createRoot(document.getElementById('root')!).render(
  <AppStateProvider>
    <App />
  </AppStateProvider>,
)
