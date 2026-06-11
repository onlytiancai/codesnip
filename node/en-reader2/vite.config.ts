import { defineConfig, loadEnv } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  return {
    plugins: [vue(), tailwindcss()],
    base: env.VITE_BASE_PATH || '/',
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
      }
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            // 将 TTS 相关的代码和依赖拆分到单独的 chunk
            'tts': ['kokoro-js'],
            // 将服务模块拆分到单独的 chunk
            'services': ['src/services/tts.ts'],
            // 将核心依赖拆分
            'vendor': ['vue']
          }
        }
      }
    }
  }
})
