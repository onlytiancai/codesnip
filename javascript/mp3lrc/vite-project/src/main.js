import { createApp } from 'vue'
import './style.css'
import weui from 'weui.js'
import 'weui'
import App from './App.vue'
import router from './router'

const app = createApp(App)
app.config.globalProperties.$weui = weui
app.use(router).mount('#app')
