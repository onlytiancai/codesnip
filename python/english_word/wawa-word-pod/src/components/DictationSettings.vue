<script setup>
import { Button } from '@/components/ui/button'

const props = defineProps({
  settings: {
    type: Object,
    required: true
  },
  isOpen: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:isOpen'])

const closeSettings = () => {
  emit('update:isOpen', false)
}
</script>

<template>
  <!-- 设置悬浮框 -->
  <div v-if="isOpen" class="fixed top-0 left-0 w-full h-full z-50 flex items-center justify-center">
    <div class="absolute inset-0 bg-black/50" @click="closeSettings"></div>
    <div class="relative bg-white dark:bg-slate-800 rounded-xl p-6 shadow-2xl w-full max-w-md">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-xl font-semibold text-slate-800 dark:text-white">听写设置</h3>
        <Button
          variant="ghost"
          @click="closeSettings"
          class="h-8 w-8 p-0"
          title="关闭"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </Button>
      </div>
      
      <div class="grid grid-cols-2 gap-6">
        <!-- 左侧设置 -->
        <div class="space-y-6">
          <!-- 单词朗读次数 -->
          <div>
            <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15.536a5 5 0 010-7.072m2.828-9.9a9 9 0 010 12.728" />
              </svg>
              播放次数
            </label>
            <div class="relative">
              <select 
                v-model="settings.repeatCount[0]" 
                class="w-full px-4 py-3 pr-10 text-sm border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all appearance-none"
              >
                <option :value="1">1次</option>
                <option :value="2">2次</option>
                <option :value="3">3次</option>
                <option :value="4">4次</option>
                <option :value="5">5次</option>
              </select>
              <div class="absolute inset-y-0 right-0 flex items-center px-3 pointer-events-none">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </div>
          </div>

          <!-- 单词停顿间隔 -->
          <div>
            <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
              播放间隔
            </label>
            <div class="relative">
              <select 
                v-model="settings.pauseBetweenWords[0]" 
                class="w-full px-4 py-3 pr-10 text-sm border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all appearance-none"
              >
                <option :value="1000">1秒</option>
                <option :value="2000">2秒</option>
                <option :value="3000">3秒</option>
                <option :value="4000">4秒</option>
                <option :value="5000">5秒</option>
                <option :value="6000">6秒</option>
                <option :value="7000">7秒</option>
                <option :value="8000">8秒</option>
                <option :value="9000">9秒</option>
                <option :value="10000">10秒</option>
              </select>
              <div class="absolute inset-y-0 right-0 flex items-center px-3 pointer-events-none">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        <!-- 右侧设置 -->
        <div class="space-y-6">
          <!-- 音频播放设置 -->
          <div>
            <h4 class="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15.536a5 5 0 010-7.072m2.828-9.9a9 9 0 010 12.728" />
              </svg>
              音频播放选项
            </h4>
            <div class="space-y-3">
              <div class="flex items-center gap-3">
                <input id="play-chinese" type="checkbox" v-model="settings.playChinese" class="h-5 w-5 text-primary-500 border-slate-300 dark:border-slate-600 rounded" />
                <label for="play-chinese" class="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2 cursor-pointer">
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15.536a5 5 0 010-7.072m2.828-9.9a9 9 0 010 12.728" />
                  </svg>
                  播放中文
                </label>
              </div>
              <div class="flex items-center gap-3">
                <input id="play-english" type="checkbox" v-model="settings.playEnglish" class="h-5 w-5 text-primary-500 border-slate-300 dark:border-slate-600 rounded" />
                <label for="play-english" class="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2 cursor-pointer">
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15.536a5 5 0 010-7.072m2.828-9.9a9 9 0 010 12.728" />
                  </svg>
                  播放英文
                </label>
              </div>
              <div class="flex items-center gap-3">
                <input id="shuffle" type="checkbox" v-model="settings.shuffle" class="h-5 w-5 text-primary-500 border-slate-300 dark:border-slate-600 rounded" />
                <label for="shuffle" class="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2 cursor-pointer">
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 3h5l-7 7-7-7h5V1h7v2zM4 15v2h7v-2H4zm0 8h7v-2H4v2zm15-8h-5v2h5v-2zm-5 8h5v-2h-5v2z" />
                  </svg>
                  随机播放
                </label>
              </div>
            </div>
          </div>

          <!-- 文本显示设置 -->
          <div>
            <h4 class="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
              文本显示选项
            </h4>
            <div class="space-y-3">
              <div class="flex items-center gap-3">
                <input id="show-english" type="checkbox" v-model="settings.showEnglish" class="h-5 w-5 text-primary-500 border-slate-300 dark:border-slate-600 rounded" />
                <label for="show-english" class="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2 cursor-pointer">
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                  显示英文
                </label>
              </div>
              <div class="flex items-center gap-3">
                <input id="show-chinese" type="checkbox" v-model="settings.showChinese" class="h-5 w-5 text-primary-500 border-slate-300 dark:border-slate-600 rounded" />
                <label for="show-chinese" class="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2 cursor-pointer">
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                  显示中文
                </label>
              </div>
              <div class="flex items-center gap-3">
                <input id="show-phonetic" type="checkbox" v-model="settings.showPhonetic" class="h-5 w-5 text-primary-500 border-slate-300 dark:border-slate-600 rounded" />
                <label for="show-phonetic" class="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2 cursor-pointer">
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                  显示音标
                </label>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div class="flex justify-end mt-6">
        <Button
          variant="default"
          @click="closeSettings"
          class="px-6 py-2.5 text-base font-medium"
        >
          确认
        </Button>
      </div>
    </div>
  </div>
</template>