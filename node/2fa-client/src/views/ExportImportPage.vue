<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 font-sans">
    <!-- 顶部导航栏 -->
    <header class="bg-white shadow-sm border-b border-gray-200">
      <div class="container mx-auto px-4 py-4 flex justify-between items-center">
        <div class="flex items-center">
          <div class="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center mr-3">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </div>
          <div>
            <h1 class="text-xl font-bold text-gray-900">2FA 客户端</h1>
            <p class="text-sm text-gray-500">安全管理您的双重认证码</p>
          </div>
        </div>
        <div class="flex items-center space-x-4">
          <router-link 
            to="/dashboard"
            class="bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 flex items-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
            返回首页
          </router-link>
          <button
            @click="handleLogout"
            class="bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-all duration-200 flex items-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l-4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
            </svg>
            退出
          </button>
        </div>
      </div>
    </header>
    
    <!-- 主要内容 -->
    <main class="container mx-auto px-4 py-8">
      <!-- 错误和成功信息 -->
      <div v-if="errorMessage" class="bg-red-50 border-l-4 border-red-500 text-red-700 p-4 rounded-r-md mb-6 shadow-sm animate-fade-in max-w-4xl mx-auto">
        <div class="flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>{{ errorMessage }}</span>
        </div>
      </div>
      <div v-if="successMessage" class="bg-green-50 border-l-4 border-green-500 text-green-700 p-4 rounded-r-md mb-6 shadow-sm animate-fade-in max-w-4xl mx-auto">
        <div class="flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
          <span>{{ successMessage }}</span>
        </div>
      </div>
      
      <div class="max-w-4xl mx-auto">
        <div class="bg-white rounded-xl shadow-sm border border-gray-100">
          <!-- 标签页导航 -->
          <div class="border-b border-gray-200">
            <nav class="flex -mb-px">
              <button
                @click="activeTab = 'export'"
                :class="[
                  'py-4 px-6 text-center font-medium text-sm border-b-2 transition-colors duration-200',
                  activeTab === 'export' 
                    ? 'border-blue-600 text-blue-600' 
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                ]"
              >
                <div class="flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  导出账户
                </div>
              </button>
              <button
                @click="activeTab = 'import'"
                :class="[
                  'py-4 px-6 text-center font-medium text-sm border-b-2 transition-colors duration-200',
                  activeTab === 'import' 
                    ? 'border-blue-600 text-blue-600' 
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                ]"
              >
                <div class="flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  导入账户
                </div>
              </button>
            </nav>
          </div>
          
          <!-- 标签页内容 -->
          <div class="p-6">
            <!-- 导出标签页 -->
            <div v-if="activeTab === 'export'">
              <div class="mb-6">
                <label for="exportText" class="block text-sm font-medium text-gray-700 mb-2">
                  账户列表（每行一个账号，格式：账户名称+发行者+加密后的2FA密钥）
                </label>
                <textarea
                  id="exportText"
                  v-model="exportText"
                  class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-3 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                  rows="10"
                  readonly
                ></textarea>
              </div>
              
              <div class="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4">
                <button
                  @click="copyToClipboard"
                  :disabled="isLoading"
                  class="flex-1 bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg flex items-center justify-center"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  复制所有信息
                </button>
                <button
                  @click="downloadBackup"
                  :disabled="isLoading"
                  class="flex-1 bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg flex items-center justify-center"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  下载备份
                </button>
              </div>
            </div>
            
            <!-- 导入标签页 -->
            <div v-if="activeTab === 'import'">
              <div v-if="!showImportConfirm" class="space-y-6">
                <div>
                  <label for="importText" class="block text-sm font-medium text-gray-700 mb-2">
                    粘贴导出的账户信息
                  </label>
                  <textarea
                    id="importText"
                    v-model="importText"
                    class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-3 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    rows="10"
                    placeholder="请粘贴从其他设备导出的账户信息..."
                  ></textarea>
                </div>
                
                <div>
                  <label for="importPassword" class="block text-sm font-medium text-gray-700 mb-2">
                    解密密码
                  </label>
                  <input
                    type="password"
                    id="importPassword"
                    v-model="importPassword"
                    class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-3 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    placeholder="请输入用于解密的密码..."
                  />
                </div>
                
                <button
                  @click="handleImport"
                  :disabled="isLoading || !importText || !importPassword"
                  class="w-full bg-purple-600 text-white py-3 px-4 rounded-lg hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg flex items-center justify-center"
                >
                  <svg v-if="!isLoading" xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  {{ isLoading ? '处理中...' : '解密并查看' }}
                </button>
              </div>
              
              <!-- 导入确认部分 -->
              <div v-else class="space-y-6">
                <h3 class="text-lg font-medium text-gray-700">导入确认</h3>
                
                <div class="overflow-x-auto">
                  <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                      <tr>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          选择
                        </th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          账户名称
                        </th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          发行者
                        </th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          状态
                        </th>
                      </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                      <tr v-for="(account, index) in importAccounts" :key="index" class="hover:bg-gray-50">
                        <td class="px-6 py-4 whitespace-nowrap">
                          <input
                            type="checkbox"
                            v-model="account.checked"
                            class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                          />
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap">
                          <div class="text-sm font-medium text-gray-900">{{ account.name }}</div>
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap">
                          <div class="text-sm text-gray-500">{{ account.issuer || '-' }}</div>
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap">
                          <span v-if="account.exists"
                            class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-yellow-100 text-yellow-800"
                          >
                            已存在
                          </span>
                          <span v-else
                            class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800"
                          >
                            新增
                          </span>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                
                <div class="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4">
                  <button
                    @click="cancelImport"
                    class="flex-1 bg-gray-600 text-white py-3 px-4 rounded-lg hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg flex items-center justify-center"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    取消
                  </button>
                  <button
                    @click="confirmImport"
                    :disabled="isLoading || !importAccounts.some(a => a.checked)"
                    class="flex-1 bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg flex items-center justify-center"
                  >
                    <svg v-if="!isLoading" xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                    </svg>
                    <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    {{ isLoading ? '保存中...' : '合并保存' }}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
    
    <!-- 密码输入弹窗组件 -->
    <PasswordInputDialog
      :visible="showPasswordDialog"
      :loading="isLoading"
      :error="passwordError"
      @submit="handlePasswordSubmit"
      @cancel="handlePasswordCancel"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { loadAccountList, saveAccountList } from '../utils/accountManager'
import { encryptData, decryptData, hasStoredAccounts } from '../utils/storage'
import { generateUUID } from '../utils/common'
import type { TwoFAAccount } from '../utils/2fa'
import PasswordInputDialog from '../components/PasswordInputDialog.vue'

const router = useRouter()
const route = useRoute()
const accounts = ref<TwoFAAccount[]>([])
const exportText = ref('')
const importText = ref('')
const importPassword = ref('')
const errorMessage = ref('')
const successMessage = ref('')
const isLoading = ref(false)
const showImportConfirm = ref(false)
const activeTab = ref('export')
const showPasswordDialog = ref(false)
const passwordError = ref('')
const currentPassword = ref('')
const importAccounts = ref<Array<{
  id?: string
  name: string
  secret: string
  issuer?: string
  checked: boolean
  exists: boolean
}>>([])
let pendingAction: (() => Promise<void>) | null = null

const handleLogout = () => {
  // 清除当前密码
  currentPassword.value = ''
  router.push('/')
}

// 生成导出文本
const generateExportText = (accounts: TwoFAAccount[], password: string): string => {
  return accounts.map(account => {
    // 加密2FA密钥
    const encryptedSecret = encryptData(account.secret, password)
    return `${account.name}|${account.issuer || ''}|${encryptedSecret}`
  }).join('\n')
}

// 解析导入文本
const parseImportText = (text: string): Array<{
  name: string
  issuer: string
  encryptedSecret: string
}> => {
  return text.split('\n')
    .filter(line => line.trim())
    .map(line => {
      const [name, issuer, encryptedSecret] = line.split('|')
      return {
        name: name || '',
        issuer: issuer || '',
        encryptedSecret: encryptedSecret || ''
      }
    })
    .filter(item => item.name && item.encryptedSecret)
}

// 密码输入弹窗相关
const handlePasswordSubmit = async (password: string) => {
  try {
    isLoading.value = true
    passwordError.value = ''
    
    // 验证密码是否正确
    if (hasStoredAccounts()) {
      try {
        // 尝试加载账户，验证密码是否正确
        const testAccounts = loadAccountList(password)
        currentPassword.value = password
        showPasswordDialog.value = false
        
        // 执行待处理的操作
        if (pendingAction) {
          await pendingAction()
        } else {
          // 如果没有待处理操作，加载账户并生成导出文本
          accounts.value = testAccounts
          exportText.value = generateExportText(accounts.value, currentPassword.value)
        }
      } catch (error) {
        passwordError.value = '密码错误，请重试。'
      }
    } else {
      // 没有存储的账户，直接保存密码
      currentPassword.value = password
      showPasswordDialog.value = false
      
      // 执行待处理的操作
      if (pendingAction) {
        await pendingAction()
      }
    }
  } catch (error) {
    passwordError.value = '验证失败，请重试。'
  } finally {
    isLoading.value = false
  }
}

const handlePasswordCancel = () => {
  showPasswordDialog.value = false
  passwordError.value = ''
}

// 加载账户并生成导出文本
onMounted(() => {
  // 检查路由参数中是否有密码
  const passwordFromRoute = route.query.password as string
  if (passwordFromRoute) {
    currentPassword.value = passwordFromRoute
    // 尝试加载账户
    try {
      accounts.value = loadAccountList(passwordFromRoute)
      exportText.value = generateExportText(accounts.value, currentPassword.value)
    } catch (error) {
      // 密码错误，显示弹窗
      showPasswordDialog.value = true
    }
  } else if (hasStoredAccounts()) {
    // 显示密码输入弹窗
    showPasswordDialog.value = true
  }
})

// 复制到剪贴板
const copyToClipboard = () => {
  navigator.clipboard.writeText(exportText.value).then(() => {
    successMessage.value = '已复制到剪贴板！'
    setTimeout(() => {
      successMessage.value = ''
    }, 3000)
  }).catch(() => {
    errorMessage.value = '复制失败，请手动复制。'
    setTimeout(() => {
      errorMessage.value = ''
    }, 3000)
  })
}

// 下载备份
const downloadBackup = () => {
  const blob = new Blob([exportText.value], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `2fa-backup-${new Date().toISOString().split('T')[0]}.txt`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
  
  successMessage.value = '备份下载成功！'
  setTimeout(() => {
    successMessage.value = ''
  }, 3000)
}

// 处理导入
const handleImport = async () => {
  try {
    isLoading.value = true
    await new Promise(resolve => setTimeout(resolve, 300))
    
    // 解析导入文本
    const parsedAccounts = parseImportText(importText.value)
    
    if (parsedAccounts.length === 0) {
      errorMessage.value = '未找到有效的账户信息，请检查输入格式。'
      setTimeout(() => {
        errorMessage.value = ''
      }, 3000)
      return
    }
    
    // 解密并验证账户信息
    const decryptedAccounts = parsedAccounts.map(item => {
      try {
        const secret = decryptData(item.encryptedSecret, importPassword.value)
        return {
          name: item.name,
          secret,
          issuer: item.issuer,
          checked: true,
          exists: accounts.value.some(account => 
            account.name === item.name && account.issuer === item.issuer
          )
        }
      } catch (error) {
        throw new Error('解密失败，请检查密码是否正确。')
      }
    })
    
    importAccounts.value = decryptedAccounts
    showImportConfirm.value = true
    successMessage.value = '解密成功，请确认要导入的账户。'
    setTimeout(() => {
      successMessage.value = ''
    }, 3000)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '导入失败，请检查输入和密码。'
    setTimeout(() => {
      errorMessage.value = ''
    }, 3000)
  } finally {
    isLoading.value = false
  }
}

// 取消导入
const cancelImport = () => {
  showImportConfirm.value = false
  importText.value = ''
  importPassword.value = ''
  importAccounts.value = []
}

// 确认导入并合并保存
const confirmImport = async () => {
  // 检查是否有当前密码，如果没有则显示密码输入框
  if (!currentPassword.value) {
    pendingAction = async () => {
      await doConfirmImport()
    }
    showPasswordDialog.value = true
    return
  }
  
  await doConfirmImport()
}

const doConfirmImport = async () => {
  try {
    isLoading.value = true
    await new Promise(resolve => setTimeout(resolve, 300))
    
    // 过滤出选中的账户
    const selectedAccounts = importAccounts.value.filter(account => account.checked)
    
    if (selectedAccounts.length === 0) {
      errorMessage.value = '请至少选择一个账户进行导入。'
      setTimeout(() => {
        errorMessage.value = ''
      }, 3000)
      return
    }
    

    // 合并账户：只添加不存在的账户
    const mergedAccounts = [...accounts.value]
    selectedAccounts.forEach(account => {
      const exists = mergedAccounts.some(existingAccount => 
        existingAccount.name === account.name && existingAccount.issuer === account.issuer
      )
      
      if (!exists) {
        mergedAccounts.push({
          id: generateUUID(),
          name: account.name,
          secret: account.secret,
          issuer: account.issuer
        })
      }
    })
    
    // 保存合并后的账户
    saveAccountList(mergedAccounts, currentPassword.value)
    accounts.value = mergedAccounts
    
    // 更新导出文本
    exportText.value = generateExportText(accounts.value, currentPassword.value)
    
    // 重置导入状态
    cancelImport()
    
    successMessage.value = `成功导入 ${selectedAccounts.length} 个账户！`
    setTimeout(() => {
      successMessage.value = ''
    }, 3000)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '保存失败，请重试。'
    setTimeout(() => {
      errorMessage.value = ''
    }, 3000)
  } finally {
    isLoading.value = false
    pendingAction = null
  }
}
</script>

<style>
/* 全局样式 */

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* 动画效果 */
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-fade-in {
  animation: fadeIn 0.3s ease-out forwards;
}
</style>