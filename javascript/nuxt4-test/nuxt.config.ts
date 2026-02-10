// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },
  modules: ['/Users/huhao/src/nuxt-auth-utils/src/module'],
  runtimeConfig: {
    myTestConfig: '',
  },
  vite: {
    server: {
      allowedHosts: [
        'tools.myapp1024.com',
      ],
    },
  },
})