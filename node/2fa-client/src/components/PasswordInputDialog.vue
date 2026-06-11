<template>
  <div v-if="visible" class="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center p-4 z-50">
    <div class="bg-white rounded-xl shadow-2xl p-6 w-full max-w-md transform transition-all duration-300 animate-fade-in">
      <div class="flex justify-between items-center mb-6">
        <h3 class="text-xl font-semibold text-gray-900 flex items-center">
          <Icon name="shield" size="h-5 w-5" class="mr-2 text-blue-600" />
          {{ title }}
        </h3>
        <button
          @click="handleCancel"
          class="text-gray-400 hover:text-gray-600 p-2 rounded-full hover:bg-gray-100 transition-colors"
        >
          <Icon name="x-circle" size="h-5 w-5" />
        </button>
      </div>
      <div class="mb-6">
        <div class="mb-4">
          <label for="passwordInput" class="block text-sm font-medium text-gray-700 mb-2">
            {{ label }}
          </label>
          <div class="relative">
            <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <Icon name="key" size="h-5 w-5" color="text-gray-400" />
            </div>
            <input
              type="password"
              id="passwordInput"
              v-model="password"
              class="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-3 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
              :placeholder="placeholder"
              @keyup.enter="handleSubmit"
            />
          </div>
        </div>
        <div v-if="error" class="bg-red-50 border-l-4 border-red-500 text-red-700 p-3 rounded-r-md mb-4">
          <div class="flex items-center">
            <Icon name="exclamation-circle" size="h-5 w-5" class="mr-2" />
            <span>{{ error }}</span>
          </div>
        </div>
      </div>
      <div class="space-y-3">
        <button
          @click="handleSubmit"
          :disabled="loading"
          class="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center"
        >
          <Icon v-if="!loading" name="logout" size="h-5 w-5" class="mr-2" />
          <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          {{ loading ? '验证中...' : confirmText }}
        </button>
        <button
          @click="handleCancel"
          class="w-full bg-gray-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-md hover:shadow-lg flex items-center justify-center"
        >
          <Icon name="x-circle" size="h-5 w-5" class="mr-2" />
          取消
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import Icon from './Icon.vue'

defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  loading: {
    type: Boolean,
    default: false
  },
  error: {
    type: String,
    default: ''
  },
  title: {
    type: String,
    default: '输入密码'
  },
  label: {
    type: String,
    default: '请输入主密码'
  },
  placeholder: {
    type: String,
    default: '请输入密码'
  },
  confirmText: {
    type: String,
    default: '确认'
  }
})

const emit = defineEmits(['submit', 'cancel'])
const password = ref('')

const handleSubmit = () => {
  emit('submit', password.value)
  password.value = ''
}

const handleCancel = () => {
  emit('cancel')
  password.value = ''
}
</script>

<style scoped>
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