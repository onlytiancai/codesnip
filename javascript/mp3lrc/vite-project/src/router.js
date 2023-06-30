import { createRouter, createWebHashHistory } from 'vue-router'

import List from './components/List.vue'
import Detail from './components/Detail.vue'

const routes = [
    { path: '/', component: List },
    { name:'detail', path: '/detail', component: Detail },
]
console.log(routes)
export default createRouter({    
    history: createWebHashHistory(),
    routes: routes,
})
