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
            to="/export-import" 
            :query="{ password: currentPassword }"
            class="bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 transition-all duration-200 flex items-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            导出导入
          </router-link>
          <router-link 
            to="/test" 
            target="_blank"
            class="bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 transition-all duration-200 flex items-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            测试页面
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
      
      <!-- 时间倒计时 -->
      <div class="mb-8 max-w-4xl mx-auto">
        <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <div class="flex justify-between items-center mb-4">
            <div class="text-sm font-medium text-gray-600 flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              验证码将在 {{ remainingTime }} 秒后刷新
            </div>
            <div class="text-xs text-gray-500">每30秒更新一次</div>
          </div>
          <div class="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div 
              class="bg-gradient-to-r from-blue-500 to-blue-600 h-full rounded-full transition-all duration-1000 ease-linear shadow-sm"
              :style="{ width: `${(remainingTime / 30) * 100}%` }"
            ></div>
          </div>
        </div>
      </div>
      
      <!-- 添加账户按钮 -->
      <div class="mb-8 max-w-4xl mx-auto">
        <div class="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4">
          <button
            @click="showAddAccountForm = !showAddAccountForm"
            class="flex-1 bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg flex items-center justify-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            {{ showAddAccountForm ? '取消' : '添加账户' }}
          </button>
          <button
            @click="showQRCodeScanner = true"
            class="flex-1 bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg flex items-center justify-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            扫描二维码
          </button>
        </div>
      </div>
      
      <!-- 添加账户表单 -->
      <div v-if="showAddAccountForm" class="mb-8 max-w-4xl mx-auto">
        <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100 animate-fade-in">
          <h3 class="text-xl font-semibold text-gray-700 flex items-center mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            添加新账户
          </h3>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label for="accountName" class="block text-sm font-medium text-gray-700 mb-2">
                账户名称
              </label>
              <input
                type="text"
                id="accountName"
                v-model="newAccountName"
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-3 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                placeholder="例如：GitHub"
              />
            </div>
            <div>
              <label for="accountSecret" class="block text-sm font-medium text-gray-700 mb-2">
                2FA 密钥
              </label>
              <input
                type="text"
                id="accountSecret"
                v-model="newAccountSecret"
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-3 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                placeholder="Base32编码的密钥"
              />
            </div>
            <div>
              <label for="accountIssuer" class="block text-sm font-medium text-gray-700 mb-2">
                发行者（可选）
              </label>
              <input
                type="text"
                id="accountIssuer"
                v-model="newAccountIssuer"
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-3 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                placeholder="例如：GitHub"
              />
            </div>
          </div>
          <div class="mt-6">
            <button
              @click="handleAddAccount"
              :disabled="isLoading"
              class="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg disabled:opacity-70 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center"
            >
              <svg v-if="!isLoading" xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
              </svg>
              <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              {{ isLoading ? '保存中...' : '保存' }}
            </button>
          </div>
        </div>
      </div>
      
      <!-- 账户列表 -->
      <div class="max-w-4xl mx-auto">
        <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100 mb-6">
          <h2 class="text-xl font-semibold text-gray-700 flex items-center mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
            我的账户
          </h2>
          
          <div v-if="accounts.length === 0" class="text-center text-gray-500 py-12 bg-gray-50 rounded-xl border border-dashed border-gray-200">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 mx-auto text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            <p class="text-gray-500">暂无账户，请添加您的第一个2FA账户。</p>
          </div>
          
          <div v-else class="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div 
              v-for="item in totpCodes" 
              :key="item.id"
              class="flex flex-col p-5 bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-md transition-all duration-200 transform hover:translate-y-[-2px] group relative"
            >
              <!-- 复制成功反馈 -->
              <div 
                v-if="showCopiedFeedback && copiedCode === item.code"
                class="absolute inset-0 bg-green-50 border-2 border-green-500 rounded-xl flex items-center justify-center animate-fade-in z-10"
              >
                <div class="flex items-center text-green-700">
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                  </svg>
                  <span class="font-medium">已复制到剪贴板！</span>
                </div>
              </div>
              
              <div class="flex items-center justify-between mb-4">
                <div class="flex items-center">
                  <div class="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center mr-3">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                  <div>
                    <div class="font-medium text-gray-900 group-hover:text-blue-600 transition-colors">{{ item.name }}</div>
                    <div v-if="item.issuer" class="text-xs text-gray-500">{{ item.issuer }}</div>
                  </div>
                </div>
                <button 
                  @click="handleRemoveAccount(item.id)"
                  :disabled="isLoading"
                  class="text-gray-400 hover:text-red-600 p-2 rounded-full hover:bg-red-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
              
              <div class="mt-auto">
                <div class="flex items-center justify-between">
                  <div class="text-3xl font-mono font-bold text-blue-600 group-hover:scale-105 transition-transform">{{ item.code }}</div>
                  <button 
                    @click="copyToClipboard(item.code)"
                    class="text-sm text-blue-500 hover:text-blue-700 inline-flex items-center transition-colors bg-blue-50 px-3 py-1 rounded-full"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    复制
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
    
    <!-- 二维码扫描器 -->
    <div v-if="showQRCodeScanner" class="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center p-4 z-50">
      <div class="bg-white rounded-xl shadow-2xl p-6 w-full max-w-md transform transition-all duration-300 animate-fade-in">
        <div class="flex justify-between items-center mb-6">
          <h3 class="text-xl font-semibold text-gray-900 flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            扫描二维码
          </h3>
          <button
            @click="showQRCodeScanner = false"
            class="text-gray-400 hover:text-gray-600 p-2 rounded-full hover:bg-gray-100 transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div class="mb-6 relative">
          <div class="aspect-video bg-gray-900 rounded-lg overflow-hidden relative">
            <video id="qrScanner" class="w-full h-full object-cover" autoplay></video>
            <!-- 扫描线动画 -->
            <div class="absolute inset-0 flex items-center justify-center">
              <div class="w-4/5 h-4/5 border-2 border-blue-500 rounded-lg animate-pulse"></div>
            </div>
            <div class="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div class="w-full h-1 bg-blue-500 opacity-75 animate-scan"></div>
            </div>
          </div>
          <div class="text-center mt-4 text-sm text-gray-500">
            请将二维码对准摄像头
          </div>
        </div>
        <div>
          <button
            @click="showQRCodeScanner = false"
            class="w-full bg-gray-600 text-white py-3 px-4 rounded-lg hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg flex items-center justify-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
            取消
          </button>
        </div>
      </div>
    </div>
    
    <!-- 密码输入弹窗 -->
    <div v-if="showPasswordDialog" class="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center p-4 z-50">
      <div class="bg-white rounded-xl shadow-2xl p-6 w-full max-w-md transform transition-all duration-300 animate-fade-in">
        <div class="flex justify-between items-center mb-6">
          <h3 class="text-xl font-semibold text-gray-900 flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            输入密码
          </h3>
          <button
            @click="showPasswordDialog = false"
            class="text-gray-400 hover:text-gray-600 p-2 rounded-full hover:bg-gray-100 transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div class="mb-6">
          <div class="mb-4">
            <label for="passwordInput" class="block text-sm font-medium text-gray-700 mb-2">
              请输入主密码
            </label>
            <div class="relative">
              <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <input
                type="password"
                id="passwordInput"
                v-model="passwordInput"
                class="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-3 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                placeholder="请输入密码"
                @keyup.enter="handlePasswordSubmit"
              />
            </div>
          </div>
          <div v-if="passwordError" class="bg-red-50 border-l-4 border-red-500 text-red-700 p-3 rounded-r-md mb-4">
            <div class="flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{{ passwordError }}</span>
            </div>
          </div>
        </div>
        <div class="space-y-3">
          <button
            @click="handlePasswordSubmit"
            :disabled="isLoading"
            class="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center"
          >
            <svg v-if="!isLoading" xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1" />
            </svg>
            <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            {{ isLoading ? '验证中...' : '确认' }}
          </button>
          <button
            @click="showPasswordDialog = false"
            class="w-full bg-gray-600 text-white py-3 px-4 rounded-lg hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg flex items-center justify-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
            取消
          </button>
        </div>
      </div>
    </div>
    
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
import { ref, onMounted, onUnmounted, computed, watchEffect } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { loadAccountList, saveAccountList, addAccount, removeAccount } from '../utils/accountManager'
import { generateTOTP, getRemainingTime } from '../utils/2fa'
import { extractSecretFromQRCode } from '../utils/qrCode'
import { hasStoredAccounts } from '../utils/storage'
import PasswordInputDialog from '../components/PasswordInputDialog.vue'

const router = useRouter()
const route = useRoute()
const accounts = ref<any[]>([])
const currentTime = ref(Date.now())
const showAddAccountForm = ref(false)
const showQRCodeScanner = ref(false)
const showPasswordDialog = ref(false)
const passwordInput = ref('')
const passwordError = ref('')
const currentPassword = ref('')
const newAccountName = ref('')
const newAccountSecret = ref('')
const newAccountIssuer = ref('')
const errorMessage = ref('')
const successMessage = ref('')
const isLoading = ref(false)
const copiedCode = ref('')
const showCopiedFeedback = ref(false)
let interval: number | undefined
let pendingAction: (() => Promise<void>) | null = null

const handleLogout = () => {
  // 清除当前密码
  currentPassword.value = ''
  router.push('/')
}

const remainingTime = computed(() => {
  currentTime.value
  return getRemainingTime()
})

const totpCodes = computed(() => {
  currentTime.value
  return accounts.value.map(account => ({
    ...account,
    code: generateTOTP(account.secret),
    remainingTime: remainingTime.value
  }))
})

const updateTime = () => {
  currentTime.value = Date.now()
}

// 保存账户
const saveAccounts = async () => {
  try {
    isLoading.value = true
    await new Promise(resolve => setTimeout(resolve, 300))
    saveAccountList(accounts.value, currentPassword.value)
    successMessage.value = '账户已保存！'
    setTimeout(() => {
      successMessage.value = ''
    }, 3000)
  } catch (err) {
    errorMessage.value = '保存失败，请重试。'
    setTimeout(() => {
      errorMessage.value = ''
    }, 3000)
  } finally {
    isLoading.value = false
  }
}

// 添加新账户
const handleAddAccount = async () => {
  if (!newAccountName.value || !newAccountSecret.value) {
    errorMessage.value = '请填写账户名称和密钥。'
    setTimeout(() => {
      errorMessage.value = ''
    }, 3000)
    return
  }
  
  // 检查是否有当前密码，如果没有则显示密码输入框
  if (!currentPassword.value) {
    pendingAction = async () => {
      try {
        isLoading.value = true
        await new Promise(resolve => setTimeout(resolve, 300))
        accounts.value = addAccount(accounts.value, newAccountName.value, newAccountSecret.value, newAccountIssuer.value)
        await saveAccounts()
        
        // 重置表单
        newAccountName.value = ''
        newAccountSecret.value = ''
        newAccountIssuer.value = ''
        showAddAccountForm.value = false
        successMessage.value = '账户添加成功！'
        setTimeout(() => {
          successMessage.value = ''
        }, 3000)
      } catch (err) {
        errorMessage.value = '添加账户失败，请重试。'
        setTimeout(() => {
          errorMessage.value = ''
        }, 3000)
      } finally {
        isLoading.value = false
        pendingAction = null
      }
    }
    showPasswordDialog.value = true
    return
  }
  
  try {
    isLoading.value = true
    await new Promise(resolve => setTimeout(resolve, 300))
    accounts.value = addAccount(accounts.value, newAccountName.value, newAccountSecret.value, newAccountIssuer.value)
    await saveAccounts()
    
    // 重置表单
    newAccountName.value = ''
    newAccountSecret.value = ''
    newAccountIssuer.value = ''
    showAddAccountForm.value = false
    successMessage.value = '账户添加成功！'
    setTimeout(() => {
      successMessage.value = ''
    }, 3000)
  } catch (err) {
    errorMessage.value = '添加账户失败，请重试。'
    setTimeout(() => {
      errorMessage.value = ''
    }, 3000)
  } finally {
    isLoading.value = false
  }
}

// 删除账户
const handleRemoveAccount = async (accountId: string) => {
  // 检查是否有当前密码，如果没有则显示密码输入框
  if (!currentPassword.value) {
    pendingAction = async () => {
      try {
        isLoading.value = true
        await new Promise(resolve => setTimeout(resolve, 300))
        accounts.value = removeAccount(accounts.value, accountId)
        await saveAccounts()
        successMessage.value = '账户已删除！'
        setTimeout(() => {
          successMessage.value = ''
        }, 3000)
      } catch (err) {
        errorMessage.value = '删除账户失败，请重试。'
        setTimeout(() => {
          errorMessage.value = ''
        }, 3000)
      } finally {
        isLoading.value = false
        pendingAction = null
      }
    }
    showPasswordDialog.value = true
    return
  }
  
  try {
    isLoading.value = true
    await new Promise(resolve => setTimeout(resolve, 300))
    accounts.value = removeAccount(accounts.value, accountId)
    await saveAccounts()
    successMessage.value = '账户已删除！'
    setTimeout(() => {
      successMessage.value = ''
    }, 3000)
  } catch (err) {
    errorMessage.value = '删除账户失败，请重试。'
    setTimeout(() => {
      errorMessage.value = ''
    }, 3000)
  } finally {
    isLoading.value = false
  }
}

// 复制TOTP码到剪贴板
const copyToClipboard = (code: string) => {
  navigator.clipboard.writeText(code).then(() => {
    copiedCode.value = code
    showCopiedFeedback.value = true
    setTimeout(() => {
      showCopiedFeedback.value = false
    }, 2000)
  }).catch(() => {
    errorMessage.value = '复制失败，请手动复制。'
    setTimeout(() => {
      errorMessage.value = ''
    }, 3000)
  })
}

// 初始化二维码扫描器
const initQRCodeScanner = () => {
  if (showQRCodeScanner.value) {
    const video = document.getElementById('qrScanner') as HTMLVideoElement
    if (video) {
      navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
        .then(stream => {
          video.srcObject = stream
          video.addEventListener('play', () => {
            const canvas = document.createElement('canvas')
            const ctx = canvas.getContext('2d')
            
            const scan = () => {
              if (!video.paused && !video.ended && ctx) {
                canvas.width = video.videoWidth
                canvas.height = video.videoHeight
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
                const result = extractSecretFromQRCode(imageData)
                
                if (result) {
                  newAccountName.value = result.name
                  newAccountSecret.value = result.secret
                  newAccountIssuer.value = result.issuer || ''
                  showQRCodeScanner.value = false
                  successMessage.value = '二维码扫描成功！'
                  
                  // 停止视频流
                  const stream = video.srcObject as MediaStream
                  if (stream) {
                    stream.getTracks().forEach(track => track.stop())
                  }
                } else {
                  requestAnimationFrame(scan)
                }
              }
            }
            
            requestAnimationFrame(scan)
          })
        })
        .catch(err => {
          console.error('无法访问摄像头:', err)
          errorMessage.value = '无法访问摄像头，请检查权限设置。'
          showQRCodeScanner.value = false
        })
    }
  }
}

// 监听二维码扫描器显示状态
watchEffect(() => {
  initQRCodeScanner()
})

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
          // 如果没有待处理操作，尝试加载账户
          accounts.value = testAccounts
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

onMounted(() => {
  // 检查路由参数中是否有密码
  const passwordFromRoute = route.query.password as string
  if (passwordFromRoute) {
    currentPassword.value = passwordFromRoute
    // 尝试加载账户
    try {
      accounts.value = loadAccountList(passwordFromRoute)
    } catch (error) {
      // 密码错误，显示弹窗
      showPasswordDialog.value = true
    }
  } else if (hasStoredAccounts()) {
    // 显示密码输入弹窗
    showPasswordDialog.value = true
  }
  
  // 启动时间更新定时器
  updateTime()
  interval = window.setInterval(updateTime, 1000)
})

onUnmounted(() => {
  if (interval) {
    clearInterval(interval)
  }
})
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

@keyframes scan {
  0% {
    transform: translateY(-100%);
  }
  100% {
    transform: translateY(100%);
  }
}

.animate-fade-in {
  animation: fadeIn 0.3s ease-out forwards;
}

.animate-scan {
  animation: scan 2s linear infinite;
}
</style>
