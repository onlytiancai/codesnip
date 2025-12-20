<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { 
  PlayCircle, PauseCircle, SkipBack, SkipForward, 
  ArrowLeft, RefreshCw, CheckCircle2, Settings
} from 'lucide-vue-next'

const props = defineProps({
  words: {
    type: Array,
    required: true
  },
  settings: {
    type: Object,
    default: () => ({
      repeatCount: [2],
      pauseBetweenWords: [3000],
      playChinese: true,
      playEnglish: false,
      showEnglish: true,
      showChinese: true,
      showPhonetic: false,
      shuffle: false
    })
  }
})

// 从localStorage加载设置
const loadSettingsFromLocalStorage = () => {
  try {
    const savedSettings = localStorage.getItem('dictationSettings')
    if (savedSettings) {
      const parsed = JSON.parse(savedSettings)
      // 确保repeatCount和pauseBetweenWords是数组格式
      if (typeof parsed.repeatCount === 'number') {
        parsed.repeatCount = [parsed.repeatCount]
      }
      if (typeof parsed.pauseBetweenWords === 'number') {
        parsed.pauseBetweenWords = [parsed.pauseBetweenWords]
      }
      return parsed
    }
  } catch (error) {
    console.error('Failed to load settings from localStorage:', error)
  }
  return props.settings
}

// 本地设置变量
const settings = ref(loadSettingsFromLocalStorage())

// 监听设置变化，保存到localStorage
watch(settings, (newSettings) => {
  try {
    localStorage.setItem('dictationSettings', JSON.stringify(newSettings))
  } catch (error) {
    console.error('Failed to save settings to localStorage:', error)
  }
}, { deep: true })

const emit = defineEmits(['finish'])



// 状态
const currentWordIndex = ref(0)
const currentRepeat = ref(0)
const isPlaying = ref(false)
const isPaused = ref(false)
const dictationComplete = ref(false)
const processedWords = ref([])
const showSettings = ref(false)
const pausedAudioTime = ref(0)

// 初始化processedWords
if (props.words.length > 0) {
  processedWords.value = [...props.words]
  if (settings.value.shuffle) {
    // 使用Fisher-Yates洗牌算法随机化单词顺序
    for (let i = processedWords.value.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[processedWords.value[i], processedWords.value[j]] = [processedWords.value[j], processedWords.value[i]]
    }
  }
}

// 音频元素
const audioElement = ref(null)

// 定时器
let playTimer = null
let pauseTimer = null

// 正则表达式用于normalize_filename函数
const normalizeFilename = (text) => {
  // 替换所有非字母数字字符为横杠
  let normalized = text.trim().replace(/[^a-zA-Z0-9]/g, '-')
  // 移除连续的横杠
  normalized = normalized.replace(/-+/g, '-')
  // 移除首尾的横杠
  normalized = normalized.replace(/^-+|-+$/g, '')
  return normalized.toLowerCase()
}

// 获取音频文件路径
const getAudioPath = (wordItem, language = 'cn') => {
  const text = wordItem.word || wordItem.phrase
  const filename = normalizeFilename(text)
  return `/audio/${filename}_${language}.mp3`
}

// 播放当前音频
const playAudio = (startTime = 0) => {
  if (audioElement.value && currentWordIndex.value < processedWords.value.length) {
    const currentWord = processedWords.value[currentWordIndex.value]
    const languagesToPlay = []
    
    // 根据设置确定要播放的语言
    if (settings.value.playChinese) {
      languagesToPlay.push('cn')
    }
    if (settings.value.playEnglish) {
      languagesToPlay.push('en')
    }
    
    // 如果没有选择任何语言，默认播放中文
    if (languagesToPlay.length === 0) {
      languagesToPlay.push('cn')
    }
    
    // 播放音频序列
    playAudioSequence(currentWord, languagesToPlay, 0, startTime)
  }
}

// 播放音频序列
const playAudioSequence = (word, languages, index, startTime = 0) => {
  if (index >= languages.length) {
    // 序列播放完成
    return handleAudioEnd()
  }
  
  const audioPath = getAudioPath(word, languages[index])
  
  // 先暂停当前音频并设置播放位置，避免竞态条件
  if (audioElement.value) {
    audioElement.value.pause()
    audioElement.value.currentTime = startTime
    // 移除之前的事件监听器
    audioElement.value.removeEventListener('ended', playAudioSequence.onAudioEnd)
    audioElement.value.removeEventListener('canplaythrough', playAudioSequence.onCanPlay)
  }
  
  // 设置新的音频源
  audioElement.value.src = audioPath
  
  // 定义事件处理函数
  playAudioSequence.onAudioEnd = () => {
    audioElement.value.removeEventListener('ended', playAudioSequence.onAudioEnd)
    audioElement.value.removeEventListener('canplaythrough', playAudioSequence.onCanPlay)
    // 短暂停顿后播放下一个语言的音频
    setTimeout(() => {
      playAudioSequence(word, languages, index + 1)
    }, 500)
  }
  
  playAudioSequence.onCanPlay = () => {
    audioElement.value.removeEventListener('canplaythrough', playAudioSequence.onCanPlay)
    // 设置播放位置
    audioElement.value.currentTime = startTime
    // 音频加载完成后再播放
    audioElement.value.play().catch(error => {
      console.error('音频播放失败:', error)
      // 播放失败时继续下一个音频
      playAudioSequence(word, languages, index + 1)
    })
  }
  
  // 监听音频事件
  audioElement.value.addEventListener('ended', playAudioSequence.onAudioEnd)
  audioElement.value.addEventListener('canplaythrough', playAudioSequence.onCanPlay)
}

// 音频播放结束处理
const handleAudioEnd = () => {
  currentRepeat.value++
  
  if (currentRepeat.value < settings.value.repeatCount[0]) {
    // 继续重复播放当前单词
    playTimer = setTimeout(() => {
      playAudio()
    }, 1500) // 每次重复之间的短暂停顿
  } else {
    // 进入下一个单词
    currentWordIndex.value++
    currentRepeat.value = 0
    
    if (currentWordIndex.value < processedWords.value.length) {
      // 在下一个单词播放前等待指定时间
      pauseTimer = setTimeout(() => {
      playAudio()
    }, settings.value.pauseBetweenWords[0])
    } else {
      // 听写完成
      isPlaying.value = false
      dictationComplete.value = true
    }
  }
}

// 开始听写
const startDictation = () => {
  if (!isPlaying.value && !dictationComplete.value && props.words.length > 0) {
    // 重置状态
    currentWordIndex.value = 0
    currentRepeat.value = 0
    dictationComplete.value = false
    
    // 处理单词列表，根据设置决定是否随机排序
    processedWords.value = [...props.words]
    if (settings.value.shuffle) {
      // 使用Fisher-Yates洗牌算法随机化单词顺序
      for (let i = processedWords.value.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[processedWords.value[i], processedWords.value[j]] = [processedWords.value[j], processedWords.value[i]]
      }
    }
    isPlaying.value = true
    isPaused.value = false
    
    // 确保音频元素存在后再播放
    if (audioElement.value) {
      playAudio()
    } else {
      // 音频元素还未准备好，等待一下再尝试
      setTimeout(() => {
        playAudio()
      }, 100)
    }
  }
}

// 暂停听写
const pauseDictation = () => {
  if (isPlaying.value) {
    isPaused.value = true
    isPlaying.value = false
    
    // 清除所有定时器
    if (playTimer) {
      clearTimeout(playTimer)
      playTimer = null
    }
    if (pauseTimer) {
      clearTimeout(pauseTimer)
      pauseTimer = null
    }
    
    // 暂停音频并记录播放位置
    if (audioElement.value) {
      pausedAudioTime.value = audioElement.value.currentTime
      audioElement.value.pause()
    }
  }
}

// 继续听写
const resumeDictation = () => {
  if (isPaused.value && !dictationComplete.value) {
    isPaused.value = false
    isPlaying.value = true
    playAudio()
  }
}

// 上一个单词
const previousWord = () => {
  if (currentWordIndex.value > 0) {
    // 暂停当前播放和定时器
    pauseDictation()
    
    // 切换到上一个单词
    currentWordIndex.value--
    currentRepeat.value = 0
    
    // 如果之前是播放状态，继续播放
    if (isPaused.value) {
      resumeDictation()
    }
  }
}

// 下一个单词
const nextWord = () => {
  if (currentWordIndex.value < processedWords.value.length - 1) {
    // 暂停当前播放和定时器
    pauseDictation()
    
    // 切换到下一个单词
    currentWordIndex.value++
    currentRepeat.value = 0
    
    // 如果之前是播放状态，继续播放
    if (isPaused.value) {
      resumeDictation()
    }
  }
}

// 重新开始听写
const restartDictation = () => {
  // 停止当前播放
  pauseDictation()
  
  // 重置状态
  currentWordIndex.value = 0
  currentRepeat.value = 0
  isPlaying.value = false
  isPaused.value = false
  dictationComplete.value = false
  
  // 开始新的听写
  startDictation()
}

// 返回单词列表
const goBack = () => {
  pauseDictation()
  emit('finish')
}

// 监听播放状态变化
watch(isPlaying, (newVal) => {
  if (!newVal) {
    // 清除所有定时器
    if (playTimer) {
      clearTimeout(playTimer)
      playTimer = null
    }
    if (pauseTimer) {
      clearTimeout(pauseTimer)
      pauseTimer = null
    }
  }
})

// 组件挂载时自动开始听写并滚动到页面顶部
onMounted(() => {
  window.scrollTo({ top: 0, behavior: 'smooth' })
  startDictation()
})

// 组件卸载时清理
onUnmounted(() => {
  pauseDictation()
})
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 p-4 md:p-6">
    <div class="max-w-2xl mx-auto">
      <Card class="overflow-hidden shadow-lg hover:shadow-xl transition-shadow duration-300 border-0 bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm relative">
        <CardHeader class="bg-gradient-to-r from-primary-500 to-primary-600 text-white flex items-center justify-between">
          <div class="flex items-center gap-3">
            <PlayCircle class="h-8 w-8" />
            <CardTitle class="text-2xl md:text-3xl font-bold">单词听写</CardTitle>
          </div>
        </CardHeader>
        <CardContent class="pt-6">
          <!-- 设置悬浮框 -->
          <div v-if="showSettings" class="fixed top-0 left-0 w-full h-full z-50 flex items-center justify-center">
            <div class="absolute inset-0 bg-black/50" @click="showSettings = false"></div>
            <div class="relative bg-white dark:bg-slate-800 rounded-xl p-6 shadow-2xl w-full max-w-md">
              <div class="flex items-center justify-between mb-4">
                <h3 class="text-xl font-semibold text-slate-800 dark:text-white">听写设置</h3>
                <Button
                  variant="ghost"
                  @click="showSettings = false"
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
                  @click="showSettings = false"
                  class="px-6 py-2.5 text-base font-medium"
                >
                  确认
                </Button>
              </div>
            </div>
          </div>
          
          <!-- 听写进度 -->
          <div class="mb-6" v-if="!dictationComplete">
            <div class="flex justify-between text-sm text-slate-700 dark:text-slate-300 mb-2">
              <div class="flex items-center gap-2">
                <span class="font-medium">当前单词:</span> {{ currentWordIndex + 1 }} / {{ processedWords.length }}
              </div>
              <div v-if="currentWordIndex < processedWords.length" class="flex items-center gap-2">
                <span class="font-medium">当前重复:</span> {{ currentRepeat + 1 }} / {{ settings.repeatCount[0] }}
              </div>
            </div>
            <div class="w-full h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden shadow-inner">
              <div 
              class="h-full bg-gradient-to-r from-primary-500 to-primary-600 transition-all duration-300 ease-in-out rounded-full shadow-md"
              :style="{ width: `${((currentWordIndex + (currentRepeat / settings.repeatCount[0])) / processedWords.length) * 100}%` }"
            ></div>
            </div>
          </div>

          <!-- 听写内容 -->
          <div class="text-center mb-8">
            <div v-if="dictationComplete" class="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/30 dark:to-green-800/30 p-8 rounded-xl border border-green-200 dark:border-green-800 text-green-800 dark:text-green-300 shadow-lg transition-all duration-300 hover:shadow-xl">
              <CheckCircle2 class="w-12 h-12 mx-auto mb-4 text-green-500 dark:text-green-400" />
              <h3 class="text-2xl font-bold mb-3">恭喜您完成听写！</h3>
              <p class="text-lg">您总共听写了 {{ processedWords.length }} 个单词/短语。</p>
            </div>
            <div v-else-if="currentWordIndex < processedWords.length" class="bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-800/50 dark:to-slate-700/50 p-8 rounded-xl border border-slate-200 dark:border-slate-700 shadow-lg transition-all duration-300 hover:shadow-xl">
              <div class="inline-block text-left max-w-full">
                <div v-if="settings.showEnglish" class="text-3xl font-bold mb-3 text-slate-800 dark:text-white">
                  {{ processedWords[currentWordIndex].word || processedWords[currentWordIndex].phrase }}
                </div>
                <div v-if="settings.showPhonetic && (processedWords[currentWordIndex].phonetic || processedWords[currentWordIndex].phonetic_uk)" class="text-sm text-slate-500 dark:text-slate-400 mb-2 italic">
                  {{ processedWords[currentWordIndex].phonetic || processedWords[currentWordIndex].phonetic_uk }}
                </div>
                <div v-if="settings.showChinese" class="text-xl text-slate-600 dark:text-slate-300">
                  {{ processedWords[currentWordIndex].chinese }}
                </div>
              </div>
            </div>
          </div>

          <!-- 控制按钮 -->
          <div class="flex flex-col items-center gap-6">
            <!-- 上一个、播放/暂停、下一个按钮 -->
            <div v-if="!dictationComplete" class="flex justify-center gap-6">
              <Button 
                variant="outline" 
                @click="previousWord"
                :disabled="currentWordIndex <= 0"
                class="group h-14 w-14 p-0 flex items-center justify-center rounded-full transition-all hover:shadow-lg hover:scale-105 hover:border-primary-400 dark:hover:border-primary-500"
                title="上一个"
              >
                <SkipBack class="h-6 w-6 group-hover:scale-110 transition-transform" />
              </Button>
              
              <Button 
                variant="outline" 
                @click="!isPlaying ? startDictation() : (isPaused ? resumeDictation() : pauseDictation())"
                class="group h-14 w-14 p-0 flex items-center justify-center rounded-full transition-all hover:shadow-lg hover:scale-105 hover:border-primary-400 dark:hover:border-primary-500"
                :title="!isPlaying ? '开始听写' : (isPaused ? '继续' : '暂停')"
              >
                <PlayCircle v-if="!isPlaying || isPaused" class="h-6 w-6 group-hover:scale-110 transition-transform" />
                <PauseCircle v-else class="h-6 w-6 group-hover:scale-110 transition-transform" />
              </Button>

              <Button 
                variant="outline" 
                @click="nextWord"
                :disabled="currentWordIndex >= processedWords.length - 1"
                class="group h-14 w-14 p-0 flex items-center justify-center rounded-full transition-all hover:shadow-lg hover:scale-105 hover:border-primary-400 dark:hover:border-primary-500"
                title="下一个"
              >
                <SkipForward class="h-6 w-6 group-hover:scale-110 transition-transform" />
              </Button>

            </div>
            
            <!-- 重新开始和返回按钮 -->
            <div v-if="!dictationComplete" class="flex flex-wrap justify-center gap-4">
              <Button 
                variant="secondary" 
                @click="restartDictation"
                class="group flex items-center gap-2 px-6 py-2.5 text-base font-medium transition-all hover:shadow-md hover:bg-yellow-100 dark:hover:bg-yellow-900/30 hover:text-yellow-600 dark:hover:text-yellow-400"
              >
                <RefreshCw class="h-4 w-4 group-hover:rotate-180 transition-transform" />
                重新开始
              </Button>
              <Button 
                variant="secondary" 
                @click="goBack"
                class="group flex items-center gap-2 px-6 py-2.5 text-base font-medium transition-all hover:shadow-md hover:bg-slate-100 dark:hover:bg-slate-700"
              >
                <ArrowLeft class="h-4 w-4 group-hover:-translate-x-1 transition-transform" />
                返回
              </Button>
            </div>
            
            <!-- 听写完成后的按钮 -->
            <div v-else class="flex flex-wrap justify-center gap-4">
              <Button 
                variant="default" 
                @click="restartDictation"
                class="group flex items-center gap-2 px-8 py-3 text-base font-semibold shadow-lg hover:bg-green-600 hover:shadow-2xl hover:scale-105 transition-all duration-300"
              >
                <RefreshCw class="h-5 w-5 group-hover:rotate-180 transition-transform" />
                再次听写
              </Button>
              <Button 
                variant="secondary" 
                @click="goBack"
                class="group flex items-center gap-2 px-8 py-3 text-base font-medium transition-all hover:shadow-md hover:bg-slate-100 dark:hover:bg-slate-700"
              >
                <ArrowLeft class="h-5 w-5 group-hover:-translate-x-1 transition-transform" />
                返回确认页
              </Button>
            </div>
          </div>
        </CardContent>
          <!-- 右下角设置按钮 -->
          <div class="absolute bottom-4 right-4">
            <Button
              variant="default"
              @click="showSettings = !showSettings"
              class="h-12 w-12 p-0 rounded-full shadow-lg hover:shadow-xl transition-all"
              title="设置"
            >
              <Settings class="h-6 w-6" />
            </Button>
          </div>
      </Card>

      <!-- 隐藏的音频元素 -->
      <audio 
        ref="audioElement"
        preload="auto"
      />
    </div>
  </div>
</template>

