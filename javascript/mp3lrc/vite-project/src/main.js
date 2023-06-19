import { createApp } from 'vue'
import './style.css'
import weui from 'weui.js'
import 'weui'
import App from './App.vue'
const app = createApp(App)
app.config.globalProperties.$weui = weui
app.mount('#app')
