import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'home',
    component: () => import('@/views/HomeView.vue')
  },
  {
    path: '/new',
    name: 'new-project',
    component: () => import('@/views/NewProjectView.vue')
  },
  {
    path: '/project/:id',
    name: 'project',
    component: () => import('@/views/ProjectView.vue'),
    props: true
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('@/views/SettingsView.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router