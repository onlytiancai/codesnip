import { createRouter, createWebHashHistory } from 'vue-router'

import HelloWorld from './components/HelloWorld.vue'
import Detail from './components/Detail.vue'

const routes = [
    { path: '/', component: HelloWorld },
    { name:'detail', path: '/detail', component: Detail },
]
console.log(routes)
export default createRouter({    
    history: createWebHashHistory(),
    routes: routes,
})
