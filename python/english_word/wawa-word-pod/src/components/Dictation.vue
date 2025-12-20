<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'

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

const emit = defineEmits(['finish'])



// 状态
const currentWordIndex = ref(0)
const currentRepeat = ref(0)
const isPlaying = ref(false)
const isPaused = ref(false)
const dictationComplete = ref(false)
const processedWords = ref([])

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
const playAudio = () => {
  if (audioElement.value && currentWordIndex.value < processedWords.value.length) {
    const currentWord = processedWords.value[currentWordIndex.value]
    const languagesToPlay = []
    
    // 根据设置确定要播放的语言
    if (props.settings.playChinese) {
      languagesToPlay.push('cn')
    }
    if (props.settings.playEnglish) {
      languagesToPlay.push('en')
    }
    
    // 如果没有选择任何语言，默认播放中文
    if (languagesToPlay.length === 0) {
      languagesToPlay.push('cn')
    }
    
    // 播放音频序列
    playAudioSequence(currentWord, languagesToPlay, 0)
  }
}

// 播放音频序列
const playAudioSequence = (word, languages, index) => {
  if (index >= languages.length) {
    // 序列播放完成
    return handleAudioEnd()
  }
  
  const audioPath = getAudioPath(word, languages[index])
  
  audioElement.value.src = audioPath
  audioElement.value.play().catch(error => {
    console.error('音频播放失败:', error)
    // 播放失败时继续下一个音频
    playAudioSequence(word, languages, index + 1)
  })
  
  // 监听当前音频播放结束
  const onAudioEnd = () => {
    audioElement.value.removeEventListener('ended', onAudioEnd)
    // 短暂停顿后播放下一个语言的音频
    setTimeout(() => {
      playAudioSequence(word, languages, index + 1)
    }, 500)
  }
  
  audioElement.value.addEventListener('ended', onAudioEnd)
}

// 音频播放结束处理
const handleAudioEnd = () => {
  currentRepeat.value++
  
  if (currentRepeat.value < props.settings.repeatCount[0]) {
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
      }, props.settings.pauseBetweenWords[0])
    } else {
      // 听写完成
      isPlaying.value = false
      dictationComplete.value = true
    }
  }
}

// 开始听写
const startDictation = () => {
  if (!isPlaying.value && !dictationComplete.value) {
    // 处理单词列表，根据设置决定是否随机排序
    processedWords.value = [...props.words]
    if (props.settings.shuffle) {
      // 使用Fisher-Yates洗牌算法随机化单词顺序
      for (let i = processedWords.value.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[processedWords.value[i], processedWords.value[j]] = [processedWords.value[j], processedWords.value[i]]
      }
    }
    isPlaying.value = true
    isPaused.value = false
    playAudio()
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
    
    // 暂停音频
    if (audioElement.value) {
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

// 组件挂载时自动开始听写
onMounted(() => {
  startDictation()
})

// 组件卸载时清理
onUnmounted(() => {
  pauseDictation()
})
</script>

<template>
  <div class="container mx-auto p-4">
    <Card class="max-w-2xl mx-auto" :class="{'bg-white dark:bg-neutral-800': true}">
      <CardHeader class="flex flex-row items-center justify-between pb-6">
        <CardTitle class="text-2xl font-bold" :class="{'text-neutral-800 dark:text-white': true}">听写界面</CardTitle>
        <Button variant="secondary" @click="goBack" class="transition-all hover:bg-gray-100 dark:hover:bg-gray-800 hover:shadow-md">返回</Button>
      </CardHeader>
      <CardContent>


        <!-- 听写进度 -->
        <div class="mb-6" v-if="!dictationComplete">
          <div class="flex justify-between text-sm text-neutral-600 dark:text-neutral-300 mb-2">
            <div>当前单词: {{ currentWordIndex + 1 }} / {{ processedWords.length }}</div>
            <div v-if="currentWordIndex < processedWords.length">
              当前重复: {{ currentRepeat + 1 }} / {{ props.settings.repeatCount[0] }}
            </div>
          </div>
          <div class="w-full h-2 bg-neutral-200 dark:bg-neutral-700 rounded-full overflow-hidden">
            <div 
              class="h-full bg-primary-500 transition-all duration-300 ease-in-out"
              :style="{ width: `${((currentWordIndex + (currentRepeat / props.settings.repeatCount[0])) / processedWords.length) * 100}%` }"
            ></div>
          </div>
        </div>

        <!-- 听写内容 -->
        <div class="text-center mb-8">
          <div v-if="dictationComplete" class="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg border border-green-200 dark:border-green-800 text-green-800 dark:text-green-300">
            <h3 class="text-xl font-bold mb-2">恭喜您完成听写！</h3>
            <p>您总共听写了 {{ processedWords.length }} 个单词/短语。</p>
          </div>
          <div v-else-if="currentWordIndex < processedWords.length" class="bg-neutral-50 dark:bg-neutral-800 p-6 rounded-lg border">
            <div class="inline-block text-left">
              <div v-if="props.settings.showEnglish" class="text-2xl font-bold mb-1" :class="{'text-neutral-800 dark:text-white': true}">
                {{ processedWords[currentWordIndex].word || processedWords[currentWordIndex].phrase }}
              </div>
              <div v-if="props.settings.showPhonetic && (processedWords[currentWordIndex].phonetic || processedWords[currentWordIndex].phonetic_uk)" class="text-sm text-neutral-500 dark:text-neutral-400 mb-1">
                {{ processedWords[currentWordIndex].phonetic || processedWords[currentWordIndex].phonetic_uk }}
              </div>
              <div v-if="props.settings.showChinese" class="text-lg text-neutral-600 dark:text-neutral-300">
                {{ processedWords[currentWordIndex].chinese }}
              </div>
            </div>
          </div>
        </div>

        <!-- 控制按钮 -->
        <div class="flex justify-center gap-4">
          <div v-if="!dictationComplete">
            <Button 
              variant="default" 
              @click="!isPlaying ? startDictation() : (isPaused ? resumeDictation() : pauseDictation())"
              class="px-6 py-2 transition-all hover:shadow-lg hover:bg-primary-600 dark:hover:bg-primary-700"
            >
              {{ !isPlaying ? '开始听写' : (isPaused ? '继续' : '暂停') }}
            </Button>
            <Button 
              variant="secondary" 
              @click="restartDictation"
              class="ml-2 transition-all hover:bg-yellow-100 dark:hover:bg-yellow-900/30 hover:text-yellow-600 dark:hover:text-yellow-400 hover:shadow-md"
            >
              重新开始
            </Button>
          </div>
          <div v-else>
            <Button 
              variant="default" 
              @click="restartDictation"
              class="px-6 py-2 transition-all hover:shadow-lg hover:bg-green-600 dark:hover:bg-green-700"
            >
              再次听写
            </Button>
            <Button 
              variant="secondary" 
              @click="goBack"
              class="ml-2 transition-all hover:bg-gray-100 dark:hover:bg-gray-800 hover:shadow-md"
            >
              返回确认页
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>

    <!-- 隐藏的音频元素 -->
    <audio 
      ref="audioElement"
      preload="auto"
    ></audio>
  </div>
</template>

