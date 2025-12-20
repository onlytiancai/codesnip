<script setup>
import { ref, onMounted, watch } from 'vue'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Slider } from '@/components/ui/slider'
import { Checkbox } from '@/components/ui/checkbox'
import { 
  PlayCircle, Settings, XCircle, ArrowLeft, Volume2, 
  VolumeX, Shuffle, Eye, EyeOff, BookOpen 
} from 'lucide-vue-next'

const props = defineProps({
  words: {
    type: Array,
    required: true
  }
})

const emit = defineEmits(['confirm', 'cancel', 'remove-word'])

// 从localStorage加载设置
const loadSettingsFromLocalStorage = () => {
  try {
    const savedSettings = localStorage.getItem('dictationSettings')
    if (savedSettings) {
      return JSON.parse(savedSettings)
    }
  } catch (error) {
    console.error('Failed to load settings from localStorage:', error)
  }
  return {
    repeatCount: [2],
    pauseBetweenWords: [3000],
    playChinese: true,
    playEnglish: false,
    showEnglish: true,
    showChinese: true,
    showPhonetic: false,
    shuffle: false
  }
}

// 设置
const settings = ref(loadSettingsFromLocalStorage())

// 监听设置变化，保存到localStorage
watch(settings, (newSettings) => {
  try {
    localStorage.setItem('dictationSettings', JSON.stringify(newSettings))
  } catch (error) {
    console.error('Failed to save settings to localStorage:', error)
  }
}, { deep: true })

// 移除单词
const removeWord = (wordId) => {
  emit('remove-word', wordId)
}

// 确认听写
const handleConfirm = () => {
  emit('confirm', settings.value)
}

// 取消听写
const handleCancel = () => {
  emit('cancel')
}

// 组件挂载时自动开始听写并滚动到页面顶部
onMounted(() => {
  window.scrollTo({ top: 0, behavior: 'smooth' })
})
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 p-4 md:p-6">

          <!-- 所选单词 -->
          <div class="mb-8">
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-lg md:text-xl font-semibold text-slate-800 dark:text-white">
                <BookOpen class="h-5 w-5 inline mr-2" />
                你已选择 {{ words.length }} 个单词
              </h3>
              <div v-if="words.length > 0" class="text-xs px-3 py-1 bg-green-100 dark:bg-green-900/30 rounded-full text-green-700 dark:text-green-300">
                数量: {{ words.length }}
              </div>
            </div>
            
            <!-- 单词标签 -->
            <div class="flex flex-wrap gap-2 max-h-80 overflow-y-auto p-2 rounded-lg bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-700">
              <div 
                v-for="(word, index) in words" 
                :key="word.uniqueId"
                class="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium shadow-sm hover:shadow-md transition-all duration-200"
                :class="{'bg-primary-500 text-white': true}"
              >
                <span>{{ word.word || word.phrase }}</span>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  class="h-6 w-6 p-0 hover:bg-red-100/80 hover:text-red-600 transition-all rounded-full"
                  @click="removeWord(word.uniqueId)"
                >
                  <XCircle class="h-4 w-4" />
                </Button>
              </div>
              <div v-if="words.length === 0" class="flex-1 text-center text-slate-400 dark:text-slate-500 py-4">
                暂无选择的单词
              </div>
            </div>
          </div>

          <!-- 设置区域 -->
          <div class="mb-8 p-6 rounded-xl bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-800/50 dark:to-slate-900/50 border border-slate-200 dark:border-slate-700 shadow-sm">
            <div class="flex items-center gap-2 mb-4">
              <Settings class="h-5 w-5 text-primary-500" />
              <h3 class="text-lg font-semibold text-slate-800 dark:text-white">听写设置</h3>
            </div>
            
            <div class="grid grid-cols-2 gap-6">
              <!-- 左侧设置 -->
              <div class="space-y-6">
                <!-- 单词朗读次数 -->
                <div>
                  <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-2">
                    <Volume2 class="h-4 w-4 text-primary-500" />
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
                    <Volume2 class="h-4 w-4 text-primary-500" />
                    音频播放选项
                  </h4>
                  <div class="space-y-3">
                    <div class="flex items-center gap-3">
                      <Checkbox id="play-chinese" v-model="settings.playChinese" class="h-5 w-5 text-primary-500 border-slate-300 dark:border-slate-600" />
                      <label for="play-chinese" class="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2 cursor-pointer">
                        <Volume2 class="h-4 w-4 text-primary-500" />
                        播放中文
                      </label>
                    </div>
                    <div class="flex items-center gap-3">
                      <Checkbox id="play-english" v-model="settings.playEnglish" class="h-5 w-5 text-primary-500 border-slate-300 dark:border-slate-600" />
                      <label for="play-english" class="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2 cursor-pointer">
                        <Volume2 class="h-4 w-4 text-primary-500" />
                        播放英文
                      </label>
                    </div>
                    <div class="flex items-center gap-3">
                      <Checkbox id="shuffle" v-model="settings.shuffle" class="h-5 w-5 text-primary-500 border-slate-300 dark:border-slate-600" />
                      <label for="shuffle" class="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2 cursor-pointer">
                        <Shuffle class="h-4 w-4 text-primary-500" />
                        随机播放
                      </label>
                    </div>
                  </div>
                </div>

                <!-- 文本显示设置 -->
                <div>
                  <h4 class="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
                    <Eye class="h-4 w-4 text-primary-500" />
                    文本显示选项
                  </h4>
                  <div class="space-y-3">
                    <div class="flex items-center gap-3">
                      <Checkbox id="show-english" v-model="settings.showEnglish" class="h-5 w-5 text-primary-500 border-slate-300 dark:border-slate-600" />
                      <label for="show-english" class="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2 cursor-pointer">
                        <Eye class="h-4 w-4 text-primary-500" />
                        显示英文
                      </label>
                    </div>
                    <div class="flex items-center gap-3">
                      <Checkbox id="show-chinese" v-model="settings.showChinese" class="h-5 w-5 text-primary-500 border-slate-300 dark:border-slate-600" />
                      <label for="show-chinese" class="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2 cursor-pointer">
                        <Eye class="h-4 w-4 text-primary-500" />
                        显示中文
                      </label>
                    </div>
                    <div class="flex items-center gap-3">
                      <Checkbox id="show-phonetic" v-model="settings.showPhonetic" class="h-5 w-5 text-primary-500 border-slate-300 dark:border-slate-600" />
                      <label for="show-phonetic" class="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2 cursor-pointer">
                        <Eye class="h-4 w-4 text-primary-500" />
                        显示音标
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- 控制按钮 -->
          <div class="flex flex-col sm:flex-row justify-between gap-4">
            <Button 
              variant="secondary" 
              @click="handleCancel" 
              class="group flex items-center gap-2 py-3 px-6 text-base font-medium hover:bg-slate-100 dark:hover:bg-slate-700 hover:shadow-md transition-all duration-300"
            >
              <ArrowLeft class="h-5 w-5 group-hover:-translate-x-1 transition-transform" />
              返回单词列表
            </Button>
            <Button 
              variant="default" 
              @click="handleConfirm"
              :disabled="words.length === 0"
              class="group flex items-center gap-2 py-3 px-8 text-base font-semibold shadow-lg"
              :class="words.length > 0 
                ? 'bg-primary-500 hover:bg-primary-600 hover:shadow-2xl hover:scale-105' 
                : 'bg-slate-300 hover:bg-slate-400 text-slate-600 cursor-not-allowed'"
            >
              <PlayCircle class="h-5 w-5 group-hover:scale-110 transition-transform" />
              开始听写
            </Button>
          </div>    

  </div>
</template>