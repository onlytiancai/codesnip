import { createApp } from 'vue'
import Root from './Root.vue'
import { router } from './router'
import { fetchDebugFlag } from './services/auth'
import { installFetchLogger, setDebugEnabled } from './utils/debug'
import './style.css'

createApp(Root).use(router).mount('#app')

// 调试日志开关跟后端共用 server/.env 的 DEBUG —— 这里问一下再装 fetch 补丁。
void fetchDebugFlag()
  .then((res) => {
    setDebugEnabled(res.enabled === true)
    if (res.enabled) installFetchLogger()
  })
  .catch(() => {
    // 后端没起来就算了，不打日志
  })
