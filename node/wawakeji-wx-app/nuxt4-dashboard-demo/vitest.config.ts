import { defineVitestConfig } from '@nuxt/test-utils/config'

export default defineVitestConfig({
  define: {
    'process.env.TEST': 'true',
  },
  test: {
    globals: true,
    environment: 'node',
    include: ['test/unit/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
  },
})