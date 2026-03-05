// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },

  modules: [
    '@nuxt/ui',
  ],

  // Tailwind CSS configuration
  tailwindcss: {
    configPath: 'tailwind.config.ts',
  },

  // Runtime configuration
  runtimeConfig: {
    // Secret keys that are only available on the server
    apiSecret: 'default-key',
    appBaseUrl: process.env.APP_BASE_URL || 'http://localhost:3000',

    // Public keys that are also exposed to the client
    public: {
      siteName: '程序员英语',
    },
  },

  // App configuration
  app: {
    head: {
      title: '程序员英语 - 提升技术英语阅读能力',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: '专为程序员设计的英语阅读学习应用' },
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
      ],
    },
  },

  // TypeScript configuration
  typescript: {
    strict: true,
  },

  // Vite configuration
  vite: {
    test: {
      globals: true,
      environment: 'happy-dom',
    },
  },

  // Server configuration
  server: {
    port: 3000,
  },

  // Nitro configuration
  nitro: {
    storage: {
      data: {
        driver: 'fs',
        base: '.data/db',
      },
    },
  },

  future: {
    compatibilityVersion: 4,
  },

  experimental: {
    inlineRouteRules: true,
  },
})
