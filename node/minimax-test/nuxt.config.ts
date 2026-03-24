export default defineNuxtConfig({
  compatibilityDate: "2025-07-15",
  devtools: { enabled: false },

  modules: [
    ['@nuxt/ui', { fonts: false }],
    '@nuxt/icon',
    'nuxt-auth-utils'
  ],

  css: ['~/assets/css/main.css'],


    ui: {
    fonts: false
  },

  nitro: {
    experimental: {
      tasks: true
    }
  },

  runtimeConfig: {
    sessionSecret: process.env.SESSION_SECRET || 'article-scraper-secret-key-change-in-production',
    redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
    captchaEnabled: process.env.CAPTCHA_ENABLED || 'false'
  },
  public: {
    captchaEnabled: process.env.CAPTCHA_ENABLED || 'true'
  }
})
