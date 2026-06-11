<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center p-4 font-sans">
    <div class="w-full max-w-md bg-white rounded-2xl shadow-xl p-8 transform transition-all duration-300 hover:shadow-2xl border border-gray-100">
      <div class="text-center mb-8">
        <div class="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
          <Icon name="shield" size="h-8 w-8" color="text-blue-600" />
        </div>
        <h1 class="text-3xl font-bold text-gray-900">2FA 客户端</h1>
        <p class="text-gray-500 mt-2">安全管理您的双重认证码</p>
        <p class="text-gray-500 mt-2">v202602301552</p>
      </div>
      
      <!-- 错误和成功信息 -->
      <div v-if="errorMessage" class="bg-red-50 border-l-4 border-red-500 text-red-700 p-4 rounded-r-md mb-6 shadow-sm animate-fade-in">
        <div class="flex items-center">
          <Icon name="exclamation-circle" size="h-5 w-5" class="mr-2" />
          <span>{{ errorMessage }}</span>
        </div>
      </div>
      <div v-if="successMessage" class="bg-green-50 border-l-4 border-green-500 text-green-700 p-4 rounded-r-md mb-6 shadow-sm animate-fade-in">
        <div class="flex items-center">
          <Icon name="check-circle" size="h-5 w-5" class="mr-2" />
          <span>{{ successMessage }}</span>
        </div>
      </div>
      
      <div class="space-y-6">
        <div>
          <label for="password" class="block text-sm font-medium text-gray-700 mb-2">
            {{ hasStoredPasswordHash() ? '输入密码' : '请设置主密码以加密敏感信息' }}
          </label>
          <div class="relative">
            <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <Icon name="key" size="h-5 w-5" color="text-gray-400" />
            </div>
            <input
              type="password"
              id="password"
              v-model="password"
              class="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-3 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
              :placeholder="hasStoredPasswordHash() ? '请输入密码' : '请设置一个安全的主密码'"
              @keyup.enter="handleLogin"
            />
          </div>
        </div>
        <button
          @click="handleLogin"
          :disabled="isLoading"
          class="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg disabled:opacity-70 disabled:cursor-not-allowed disabled:hover:scale-100"
        >
          <div class="flex items-center justify-center">
            <Icon v-if="!isLoading" name="logout" size="h-5 w-5" class="mr-2" />
            <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            {{ isLoading ? '登录中...' : '登录' }}
          </div>
        </button>
        
        <!-- 提示信息 -->
        <div class="mt-8 space-y-3 text-sm text-gray-600">
          <div class="flex items-start">
            <Icon name="shield" size="h-4 w-4" color="text-blue-500" class="mt-0.5 mr-2 flex-shrink-0" />
            <p>本应用使用纯浏览器实现动态口令管理，所有数据均存储在本地，不会向任何服务端传递数据，代码完全开源</p>
          </div>
          <div class="flex items-start">
            <Icon name="shield" size="h-4 w-4" color="text-blue-500" class="mt-0.5 mr-2 flex-shrink-0" />
            <p>动态口令的秘钥支持加密导出到本地，您可以自行备份并在多设备间同步使用</p>
          </div>
          <div class="flex items-start">
            <Icon name="shield" size="h-4 w-4" color="text-blue-500" class="mt-0.5 mr-2 flex-shrink-0" />
            <p>首次使用时需要设置主密码，请务必牢记，后续所有加密解密操作都需要使用此密码</p>
          </div>
          <!-- GitHub 链接和安装PWA链接 -->
          <div class="mt-4 pt-4 border-t border-gray-200 flex justify-center space-x-6">
            <a 
              href="https://github.com/onlytiancai/codesnip/tree/master/node/2fa-client" 
              target="_blank" 
              rel="noopener noreferrer"
              class="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors duration-200"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16">
  <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"/>
</svg>
              <span>查看源码</span>
            </a>
            

            
            <!-- 检查更新链接（仅在PWA模式下显示） -->
            <button
              v-if="isPwaMode"
              @click="checkForUpdates"
              class="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors duration-200"
            >
              <Icon name="refresh" size="h-4 w-4" />
              <span>检查更新</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { hasStoredAccounts, loadAccounts, hasStoredPasswordHash, verifyPasswordHash, storePasswordHash } from '../utils/storage'
import { useAuthStore } from '../stores/auth'
import Icon from '../components/Icon.vue'

const router = useRouter()
const authStore = useAuthStore()
const { setPassword } = authStore
const password = ref('')
const errorMessage = ref('')
const successMessage = ref('')
const isLoading = ref(false)
// 检查更新
const checkForUpdates = async () => {
  if ('serviceWorker' in navigator) {
    try {
      const registration = await navigator.serviceWorker.ready
      await registration.update()
      
      // 检查是否有新的service worker正在等待激活
      if (registration.waiting) {
        if (confirm('有新版本可用，是否立即更新？')) {
          registration.waiting.postMessage({ type: 'SKIP_WAITING' })
          window.location.reload()
        }
      } else if (registration.installing) {
        // 监听新的service worker安装完成
        registration.installing.addEventListener('statechange', () => {
          if (registration.waiting) {
            if (confirm('有新版本可用，是否立即更新？')) {
              registration.waiting.postMessage({ type: 'SKIP_WAITING' })
              window.location.reload()
            }
          }
        })
      } else {
        successMessage.value = '当前已是最新版本！'
        setTimeout(() => {
          successMessage.value = ''
        }, 3000)
      }
    } catch (error) {
      errorMessage.value = '检查更新失败，请稍后重试。'
      setTimeout(() => {
        errorMessage.value = ''
      }, 3000)
    }
  } else {
    errorMessage.value = '当前浏览器不支持PWA更新功能。'
    setTimeout(() => {
      errorMessage.value = ''
    }, 3000)
  }
}

// 检测是否在PWA模式下运行
const isPwaMode = computed(() => {
  return window.matchMedia('(display-mode: standalone)').matches || 
         (window.navigator as any).standalone === true
})

const handleLogin = async () => {
  try {
    isLoading.value = true
    errorMessage.value = ''
    
    // 检查是否是第一次登录
    const hasPasswordHash = hasStoredPasswordHash()
    
    if (!hasPasswordHash) {
      // 第一次登录，保存密码hash
      storePasswordHash(password.value)
      successMessage.value = '密码设置成功！'
      
      // 保存密码到状态管理
      setPassword(password.value)
      
      // 延迟跳转，让用户看到成功信息
      setTimeout(() => {
        router.push({ path: '/dashboard' })
      }, 1000)
    } else {
      // 后续登录，验证密码
      const storedHash = localStorage.getItem('2fa_password_hash')
      if (!storedHash || !verifyPasswordHash(password.value, storedHash)) {
        errorMessage.value = '密码错误，请重试。'
        return
      }
      
      // 密码正确，检查是否有存储的账户
      if (hasStoredAccounts()) {
        try {
          // 尝试加载账户，验证密码是否正确
          loadAccounts(password.value)
          // 保存密码到状态管理
          setPassword(password.value)
          router.push({ path: '/dashboard' })
        } catch (error) {
          errorMessage.value = '密码错误，请重试。'
        }
      } else {
        // 没有存储的账户，直接跳转
        // 保存密码到状态管理
        setPassword(password.value)
        router.push({ path: '/dashboard' })
      }
    }
  } catch (error) {
    errorMessage.value = '登录失败，请重试。'
  } finally {
    isLoading.value = false
  }
}
</script>
