import { createApp } from 'vue'
import { createPinia } from 'pinia'
import piniaPersist from 'pinia-plugin-persistedstate'
import { router } from './router'
import './style.css'
import App from './App.vue'

const pinia = createPinia()
pinia.use(piniaPersist)

createApp(App)
  .use(pinia)
  .use(router)
  .mount('#app')
