<script setup lang="ts">
import { ref, onMounted } from 'vue';

// 状态管理
const secret = ref('');
const otpauthUrl = ref('');
const qrCodeUrl = ref('');
const userCode = ref('');
const verificationResult = ref('');
const verificationMessage = ref('');
const isLoading = ref(false);

// 复制到剪贴板的函数
const copyToClipboard = (text: string) => {
  if (typeof window !== 'undefined' && window.navigator && window.navigator.clipboard) {
    window.navigator.clipboard.writeText(text);
  }
};

// 生成新的2FA密钥
const generateSecret = async () => {
  try {
    isLoading.value = true;
    
    // 调用服务端API生成密钥
    const response = await fetch('http://localhost:3001/api/2fa/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ name: 'Test 2FA Account' })
    });
    
    if (!response.ok) {
      throw new Error('API请求失败');
    }
    
    const data = await response.json();
    
    if (data.success) {
      secret.value = data.data.secret;
      otpauthUrl.value = data.data.otpauthUrl;
      
      // 生成二维码URL
      const encodedUrl = encodeURIComponent(data.data.otpauthUrl);
      qrCodeUrl.value = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodedUrl}`;
      
      // 存储到localStorage
      localStorage.setItem('test2fa_secret', data.data.secret);
      localStorage.setItem('test2fa_otpauth', data.data.otpauthUrl);
      
      verificationResult.value = '';
      verificationMessage.value = '';
      userCode.value = '';
    } else {
      throw new Error(data.message || '生成密钥失败');
    }
    
  } catch (error) {
    console.error('生成密钥失败:', error);
    verificationResult.value = 'error';
    verificationMessage.value = '生成密钥失败，请重试。';
  } finally {
    isLoading.value = false;
  }
};

// 验证OTP
const verifyCode = async () => {
  if (!userCode.value || !secret.value) {
    verificationResult.value = 'error';
    verificationMessage.value = '请输入验证码并确保已生成密钥。';
    return;
  }
  
  try {
    isLoading.value = true;
    
    // 确保secret是字符串类型
    const secretStr = String(secret.value);
    const tokenStr = String(userCode.value);
    
    // 调用服务端API验证OTP
    const response = await fetch('http://localhost:3001/api/2fa/verify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ secret: secretStr, token: tokenStr })
    });
    
    if (!response.ok) {
      throw new Error('API请求失败');
    }
    
    const data = await response.json();
    
    if (data.success) {
      if (data.data.valid) {
        verificationResult.value = 'success';
        verificationMessage.value = '验证码验证成功！';
      } else {
        verificationResult.value = 'error';
        verificationMessage.value = '验证码验证失败，请重试。';
      }
    } else {
      throw new Error(data.message || '验证失败');
    }
    
  } catch (error) {
    console.error('验证失败:', error);
    verificationResult.value = 'error';
    verificationMessage.value = '验证失败，请重试。';
  } finally {
    isLoading.value = false;
  }
};

// 从localStorage加载数据
const loadFromStorage = () => {
  const storedSecret = localStorage.getItem('test2fa_secret');
  const storedOtpauth = localStorage.getItem('test2fa_otpauth');
  
  if (storedSecret && storedOtpauth) {
    secret.value = storedSecret;
    otpauthUrl.value = storedOtpauth;
    
    // 生成二维码URL
    const encodedUrl = encodeURIComponent(storedOtpauth);
    qrCodeUrl.value = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodedUrl}`;
  }
};

// 初始化
onMounted(() => {
  loadFromStorage();
  
  // 如果没有存储的密钥，生成一个新的
  if (!secret.value) {
    generateSecret();
  }
});
</script>

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
            <h1 class="text-xl font-bold text-gray-900">2FA 测试页面</h1>
            <p class="text-sm text-gray-500">使用 speakeasy 生成密钥和验证 OTP</p>
          </div>
        </div>
        <button 
          @click="$emit('go-back')" 
          class="bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-all duration-200 flex items-center"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
          返回首页
        </button>
      </div>
    </header>
    
    <!-- 主要内容 -->
    <main class="container mx-auto px-4 py-8">
      <div class="max-w-4xl mx-auto">
        <!-- 生成密钥部分 -->
        <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100 mb-8">
          <h2 class="text-xl font-semibold text-gray-700 flex items-center mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            生成 2FA 密钥
          </h2>
          
          <div class="space-y-6">
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                2FA 密钥 (Base32)
              </label>
              <div class="relative">
                <input
                  type="text"
                  :value="secret"
                  readonly
                  class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-3 focus:ring-blue-500 focus:border-transparent transition-all duration-200 bg-gray-50"
                />
                <button
                  @click="copyToClipboard(secret)"
                  class="absolute right-2 top-1/2 transform -translate-y-1/2 bg-blue-100 text-blue-700 px-3 py-1 rounded-md text-sm hover:bg-blue-200 transition-colors"
                >
                  复制
                </button>
              </div>
            </div>
            
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                OTP Auth URL
              </label>
              <input
                type="text"
                :value="otpauthUrl"
                readonly
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-3 focus:ring-blue-500 focus:border-transparent transition-all duration-200 bg-gray-50 font-mono text-sm"
              />
            </div>
            
            <div class="flex justify-center">
              <button
                @click="generateSecret"
                :disabled="isLoading"
                class="bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg flex items-center"
              >
                <svg v-if="!isLoading" xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
                <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                {{ isLoading ? '生成中...' : '生成新密钥' }}
              </button>
            </div>
          </div>
        </div>
        
        <!-- 二维码部分 -->
        <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100 mb-8">
          <h2 class="text-xl font-semibold text-gray-700 flex items-center mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            扫描二维码
          </h2>
          
          <div class="flex justify-center">
            <div v-if="qrCodeUrl" class="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
              <img :src="qrCodeUrl" alt="2FA QR Code" class="w-64 h-64" />
              <p class="text-center text-sm text-gray-500 mt-4">
                使用认证器应用扫描此二维码
              </p>
            </div>
            <div v-else class="w-64 h-64 bg-gray-100 rounded-lg flex items-center justify-center">
              <p class="text-gray-500">生成二维码中...</p>
            </div>
          </div>
        </div>
        
        <!-- 验证OTP部分 -->
        <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <h2 class="text-xl font-semibold text-gray-700 flex items-center mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            验证 OTP 码
          </h2>
          
          <div class="space-y-6">
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                输入认证器生成的验证码
              </label>
              <input
                type="text"
                v-model="userCode"
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-3 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                placeholder="6位数字验证码"
                maxlength="6"
                pattern="[0-9]{6}"
              />
            </div>
            
            <div class="flex justify-center">
              <button
                @click="verifyCode"
                :disabled="isLoading"
                class="bg-green-600 text-white py-3 px-6 rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg flex items-center"
              >
                <svg v-if="!isLoading" xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                </svg>
                <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                {{ isLoading ? '验证中...' : '验证' }}
              </button>
            </div>
            
            <!-- 验证结果 -->
            <div v-if="verificationResult" class="mt-4">
              <div 
                :class="[
                  'p-4 rounded-lg flex items-center',
                  verificationResult === 'success' ? 'bg-green-50 border border-green-200 text-green-700' : 'bg-red-50 border border-red-200 text-red-700'
                ]"
              >
                <svg 
                  :class="[
                    'h-5 w-5 mr-2',
                    verificationResult === 'success' ? 'text-green-500' : 'text-red-500'
                  ]"
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path 
                    :d="
                      verificationResult === 'success' 
                        ? 'M5 13l4 4L19 7' 
                        : 'M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z'
                    " 
                    stroke-linecap="round" 
                    stroke-linejoin="round" 
                    stroke-width="2"
                  />
                </svg>
                <span>{{ verificationMessage }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>