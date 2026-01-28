import { createRouter, createWebHistory } from 'vue-router'
import LoginPage from '../views/LoginPage.vue'
import Dashboard from '../views/Dashboard.vue'
import TestPage from '../views/TestPage.vue'
import ExportImportPage from '../views/ExportImportPage.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'login',
      component: LoginPage
    },
    {
      path: '/dashboard',
      name: 'dashboard',
      component: Dashboard
    },
    {
      path: '/test',
      name: 'test',
      component: TestPage
    },
    {
      path: '/export-import',
      name: 'export-import',
      component: ExportImportPage
    }
  ]
})

export default router
