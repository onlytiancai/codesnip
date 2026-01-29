<template>
  <div v-if="showInstallPrompt" class="fixed bottom-4 right-4 z-50">
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 max-w-sm">
      <div class="flex items-start space-x-3">
        <div class="flex-shrink-0">
          <svg class="h-6 w-6 text-blue-600 dark:text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <div class="flex-1">
          <h3 class="text-sm font-medium text-gray-900 dark:text-white">安装 2FA 客户端</h3>
          <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">将应用添加到您的桌面，离线使用更加便捷。</p>
        </div>
        <button
          @click="dismissPrompt"
          class="text-gray-400 hover:text-gray-500 dark:text-gray-500 dark:hover:text-gray-400"
        >
          <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <div class="flex space-x-3 mt-4">
        <button
          @click="installApp"
          class="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium py-2 px-4 rounded-md"
        >
          安装
        </button>
        <button
          @click="dismissPrompt"
          class="flex-1 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-white text-sm font-medium py-2 px-4 rounded-md"
        >
          稍后
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const showInstallPrompt = ref(false)
let deferredPrompt: any = null

const emit = defineEmits<{
  (e: 'install-success'): void
  (e: 'install-error', error: Error): void
}>()

const installApp = async () => {
  if (!deferredPrompt) return
  
  deferredPrompt.prompt()
  const { outcome } = await deferredPrompt.userChoice
  
  if (outcome === 'accepted') {
    emit('install-success')
  } else {
    emit('install-error', new Error('用户取消安装'))
  }
  
  showInstallPrompt.value = false
  deferredPrompt = null
}

const dismissPrompt = () => {
  showInstallPrompt.value = false
  localStorage.setItem('pwaInstallDismissed', 'true')
}

const handleBeforeInstallPrompt = (e: Event) => {
  e.preventDefault()
  deferredPrompt = e
  
  const dismissed = localStorage.getItem('pwaInstallDismissed')
  if (!dismissed) {
    showInstallPrompt.value = true
  }
}

const handleAppInstalled = () => {
  deferredPrompt = null
  showInstallPrompt.value = false
}

onMounted(() => {
  window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt)
  window.addEventListener('appinstalled', handleAppInstalled)
})

onUnmounted(() => {
  window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt)
  window.removeEventListener('appinstalled', handleAppInstalled)
})
</script>