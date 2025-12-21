<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { 
  PlayCircle, PauseCircle, SkipBack, SkipForward, 
  ArrowLeft, RefreshCw, CheckCircle2, Settings
} from 'lucide-vue-next'
import DictationSettings from './DictationSettings.vue'

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

         <!-- 听写设置组件 -->
          <DictationSettings :settings="settings" :is-open="showSettings" @update:is-open="showSettings = $event" />
          
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
                @click="isPaused ? resumeDictation() : pauseDictation()"
                class="group h-14 w-14 p-0 flex items-center justify-center rounded-full transition-all hover:shadow-lg hover:scale-105 hover:border-primary-400 dark:hover:border-primary-500"
                :title="isPaused ? '继续' : '暂停'"
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
      <!-- 隐藏的音频元素 -->
      <audio 
        ref="audioElement"
        preload="auto"
      />
    </div>
  </div>
</template>

