import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// three 精确匹配别名 → WebGPU 构建。
// 用正则只匹配裸 "three"，避免破坏 "three/addons/*" 前缀；
// addons 内部的 `import ... from 'three'` 也会被重定向到 WebGPU 构建，
// 保证 bundle 里只有一份 three 核心（否则类身份错乱、体积翻倍）。
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [{ find: /^three$/, replacement: 'three/webgpu' }],
  },
  build: {
    target: 'es2022',
  },
})
