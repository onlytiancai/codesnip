import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    globals: true,
    environment: 'happy-dom',
    include: ['**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
    exclude: ['**/node_modules/**', '**/dist/**', '**/.nuxt/**'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['**/*.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
      exclude: ['**/*.d.ts', '**/*.config.*', '**/mocks/**', '**/node_modules/**'],
    },
  },
})
