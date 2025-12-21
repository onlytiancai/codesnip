<script setup>
import { ref } from 'vue'
import { Button } from '@/components/ui/button'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { CheckSquare, ListChecks } from 'lucide-vue-next'

const props = defineProps({
  currentTextbook: {
    type: Object,
    required: true
  },
  textbooks: {
    type: Array,
    required: true
  },
  isDarkMode: {
    type: Boolean,
    required: true
  },
  onToggleTheme: {
    type: Function,
    required: true
  }
})

const emit = defineEmits(['switch-textbook'])

// 课本切换弹窗状态
const showTextbookModal = ref(false)

// 处理选择课本
const handleSelectTextbook = (textbook) => {
  if (!textbook.available) {
    return
  }
  
  // 如果选择的是当前课本，则直接关闭弹窗
  if (textbook.id === props.currentTextbook.id) {
    showTextbookModal.value = false
    return
  }
  
  // 发送切换课本事件
  emit('switch-textbook', textbook)
  
  // 关闭弹窗
  showTextbookModal.value = false
}
</script>

<template>
  <div class="mt-4 mb-4 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg text-purple-700 dark:text-purple-300 text-sm flex items-center justify-between shadow-md">
    <div class="flex items-center gap-2">
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-purple-500">
        <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" />
      </svg>
      <span class="font-semibold text-purple-800 dark:text-purple-200">{{ currentTextbook.name }}</span>
    </div>
    <div class="flex items-center gap-2">
      <Button 
        variant="secondary" 
        @click="showTextbookModal = true"
        class="transition-all hover:bg-purple-100 hover:text-purple-600 dark:hover:bg-purple-900/30 dark:hover:text-purple-400 hover:shadow-md"
      >
        <ListChecks class="h-3.5 w-3.5 mr-1" />
        切换
      </Button>
      <ThemeToggle :is-dark-mode="isDarkMode" :on-toggle="onToggleTheme" />
    </div>
  </div>

  <!-- 课本选择弹窗 -->
  <div 
    v-if="showTextbookModal" 
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
    @click.self="showTextbookModal = false"
  >
    <div class="bg-white dark:bg-slate-800 rounded-xl shadow-2xl max-w-md w-full max-h-[80vh] overflow-y-auto">
      <div class="p-6 border-b dark:border-slate-700">
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">选择课本</h3>
      </div>
      <div class="p-4 space-y-2">
        <div 
          v-for="textbook in textbooks" 
          :key="textbook.id"
          class="group px-4 py-3 rounded-lg cursor-pointer transition-all duration-200 flex items-center justify-between"
          :class="[
            textbook.id === currentTextbook.id 
              ? 'bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800' 
              : 'bg-slate-50 dark:bg-slate-700/50 hover:bg-slate-100 dark:hover:bg-slate-700 border border-transparent',
            !textbook.available ? 'opacity-70 cursor-not-allowed' : ''
          ]"
          @click="handleSelectTextbook(textbook)"
        >
          <div>
            <div class="font-medium text-slate-900 dark:text-white">{{ textbook.name }}</div>
            <div v-if="!textbook.available" class="text-xs text-amber-500 dark:text-amber-400 mt-1">
              敬请期待
            </div>
          </div>
          <div v-if="textbook.id === currentTextbook.id" class="text-primary-500">
            <CheckSquare class="h-5 w-5" />
          </div>
        </div>
      </div>
      <div class="p-6 border-t dark:border-slate-700 flex justify-end">
        <Button 
          variant="secondary" 
          @click="showTextbookModal = false"
        >
          关闭
        </Button>
      </div>
    </div>
  </div>
</template>