import { defineConfig, loadEnv } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const subDirectory = env.VITE_APP_SUB_DIRECTORY || 'en-speaker'
  
  return {
    plugins: [vue(), tailwindcss()],
    base: `/${subDirectory}/`,
    resolve: {
      alias: {
        '@': resolve(__dirname, 'src')
      }
    }
  }
})
