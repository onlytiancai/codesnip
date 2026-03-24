export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },

  future: {
    compatibilityVersion: 4
  },

  modules: ['@nuxt/ui', 'nuxt-auth-utils'],

  css: ['~/assets/css/main.css'],

  ui: {
    version: '4'
  },

  nitro: {
    experimental: {
      tasks: true
    }
  },

  runtimeConfig: {
    sessionSecret: process.env.SESSION_SECRET || 'article-scraper-secret-key-change-in-production',
    redisUrl: process.env.REDIS_URL || 'redis://localhost:6379'
  }
})
