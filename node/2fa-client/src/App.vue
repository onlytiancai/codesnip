<script setup lang="ts">
// 路由配置已在router/index.ts中定义
import PwaInstallPrompt from './components/PwaInstallPrompt.vue'

const handleInstallSuccess = () => {
  console.log('PWA 安装成功')
}

const handleInstallError = (error: Error) => {
  console.log('PWA 安装失败:', error)
}
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 font-sans">
    <router-view />
    <PwaInstallPrompt 
      @install-success="handleInstallSuccess"
      @install-error="handleInstallError"
    />
  </div>
</template>

