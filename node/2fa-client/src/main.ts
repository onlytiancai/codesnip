import { createApp } from 'vue'
import { createPinia } from 'pinia'
import './style.css'
import App from './App.vue'
import router from './router'

const app = createApp(App)
const pinia = createPinia()

app.use(router)
app.use(pinia)
app.mount('#app')

// PWA 更新检测
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.ready.then(registration => {
      // 监听更新
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing
        if (newWorker) {
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              // 有新版本可用
              if (confirm('有新版本可用，是否立即更新？')) {
                window.location.reload()
              }
            }
          })
        }
      })
      
      // 定期检查更新
      setInterval(() => {
        registration.update()
      }, 1000 * 60 * 60) // 每小时检查一次
    })
  })
}
