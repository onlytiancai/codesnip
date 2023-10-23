import { createRouter, createWebHashHistory } from 'vue-router'

import List from './components/List.vue'
import Home from './components/Home.vue'
import Detail from './components/Detail.vue'

const routes = [
    { path: '/', component: Home },
    { name: 'list', path: '/list', component: List },
    { name:'detail', path: '/detail', component: Detail },
]
console.log(routes)
export default createRouter({    
    history: createWebHashHistory(),
    routes: routes,
})
