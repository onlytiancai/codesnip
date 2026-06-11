import { defineVitestConfig } from '@nuxt/test-utils/config'

export default defineVitestConfig({
  define: {
    'process.env.TEST': 'true',
  },
  test: {
    globals: true,
    environment: 'node',
    include: ['test/e2e/api/**/*.test.ts'],
    setupFiles: ['./test/setup-e2e.ts'],
    testTimeout: 30000,
    hookTimeout: 30000,
    // Single thread to avoid database conflicts
    fileParallelism: false,
  },
})