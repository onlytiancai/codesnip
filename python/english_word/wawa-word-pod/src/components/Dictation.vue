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
  }
})

const emit = defineEmits(['finish'])

// 设置
const settings = ref({
  repeatCount: 2,
  pauseBetweenWords: 3000
})

// 状态
const currentWordIndex = ref(0)
const currentRepeat = ref(0)
const isPlaying = ref(false)
const isPaused = ref(false)
const dictationComplete = ref(false)

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
const getAudioPath = (wordItem) => {
  const text = wordItem.word || wordItem.phrase
  const filename = normalizeFilename(text)
  return `/audio/${filename}_cn.mp3`
}

// 播放当前中文音频
const playAudio = () => {
  if (audioElement.value && currentWordIndex.value < props.words.length) {
    const currentWord = props.words[currentWordIndex.value]
    const audioPath = getAudioPath(currentWord)
    
    audioElement.value.src = audioPath
    audioElement.value.play().catch(error => {
      console.error('音频播放失败:', error)
      // 自动继续到下一个播放
      handleAudioEnd()
    })
  }
}

// 音频播放结束处理
const handleAudioEnd = () => {
  currentRepeat.value++
  
  if (currentRepeat.value < settings.value.repeatCount) {
    // 继续重复播放当前单词
    playTimer = setTimeout(() => {
      playAudio()
    }, 500) // 每次重复之间的短暂停顿
  } else {
    // 进入下一个单词
    currentWordIndex.value++
    currentRepeat.value = 0
    
    if (currentWordIndex.value < props.words.length) {
      // 在下一个单词播放前等待指定时间
      pauseTimer = setTimeout(() => {
        playAudio()
      }, settings.value.pauseBetweenWords)
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

// 组件卸载时清理
onUnmounted(() => {
  pauseDictation()
})
</script>

<template>
  <div class="container mx-auto p-4">
    <Card class="max-w-2xl mx-auto">
      <CardHeader class="flex flex-row items-center justify-between pb-6">
        <CardTitle class="text-2xl font-bold text-neutral-800">听写界面</CardTitle>
        <Button variant="secondary" @click="goBack">返回</Button>
      </CardHeader>
      <CardContent>
        <!-- 设置区域 -->
        <div class="bg-neutral-50 p-4 rounded-lg border mb-6" v-if="!isPlaying && !dictationComplete">
          <h3 class="text-lg font-semibold mb-4 text-neutral-800">听写设置</h3>
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label for="repeatCount" class="block text-sm font-medium text-neutral-700 mb-1">每个单词读取次数:</label>
              <Input 
                id="repeatCount" 
                type="number" 
                v-model.number="settings.repeatCount" 
                min="1" 
                max="5"
                class="w-full"
              />
            </div>
            <div>
              <label for="pauseBetweenWords" class="block text-sm font-medium text-neutral-700 mb-1">单词间停顿(毫秒):</label>
              <Input 
                id="pauseBetweenWords" 
                type="number" 
                v-model.number="settings.pauseBetweenWords" 
                min="1000" 
                max="10000"
                step="500"
                class="w-full"
              />
            </div>
          </div>
        </div>

        <!-- 听写进度 -->
        <div class="mb-6" v-if="!dictationComplete">
          <div class="flex justify-between text-sm text-neutral-600 mb-2">
            <div>当前单词: {{ currentWordIndex + 1 }} / {{ words.length }}</div>
            <div v-if="currentWordIndex < words.length">
              当前重复: {{ currentRepeat + 1 }} / {{ settings.repeatCount }}
            </div>
          </div>
          <div class="w-full h-2 bg-neutral-200 rounded-full overflow-hidden">
            <div 
              class="h-full bg-green-500 transition-all duration-300 ease-in-out"
              :style="{ width: `${((currentWordIndex + (currentRepeat / settings.repeatCount)) / words.length) * 100}%` }"
            ></div>
          </div>
        </div>

        <!-- 听写内容 -->
        <div class="text-center mb-8">
          <div v-if="dictationComplete" class="bg-green-50 p-6 rounded-lg border border-green-200 text-green-800">
            <h3 class="text-xl font-bold mb-2">恭喜您完成听写！</h3>
            <p>您总共听写了 {{ words.length }} 个单词/短语。</p>
          </div>
          <div v-else-if="currentWordIndex < words.length" class="bg-neutral-50 p-6 rounded-lg border">
            <div class="inline-block text-left">
              <div class="text-2xl font-bold text-neutral-800 mb-2">
                {{ words[currentWordIndex].word || words[currentWordIndex].phrase }}
              </div>
              <div class="text-lg text-neutral-600">
                {{ words[currentWordIndex].chinese }}
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
              class="px-6 py-2"
            >
              {{ !isPlaying ? '开始听写' : (isPaused ? '继续' : '暂停') }}
            </Button>
            <Button 
              variant="secondary" 
              @click="restartDictation"
              class="ml-2"
            >
              重新开始
            </Button>
          </div>
          <div v-else>
            <Button 
              variant="default" 
              @click="restartDictation"
              class="px-6 py-2"
            >
              再次听写
            </Button>
            <Button 
              variant="secondary" 
              @click="goBack"
              class="ml-2"
            >
              返回单词列表
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>

    <!-- 隐藏的音频元素 -->
    <audio 
      ref="audioElement"
      @ended="handleAudioEnd"
      preload="auto"
    ></audio>
  </div>
</template>

