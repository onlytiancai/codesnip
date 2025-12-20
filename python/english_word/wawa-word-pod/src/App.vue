<script setup>
import { ref, onMounted, computed } from 'vue'
import WordList from './components/WordList.vue'
import Dictation from './components/Dictation.vue'
import DictationConfirm from './components/DictationConfirm.vue'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

const wordData = ref({})
const selectedWords = ref([])
const isDictationActive = ref(false)
const isConfirmActive = ref(false)
const isDarkMode = ref(false)
const dictationSettings = ref({
  repeatCount: [2],
  pauseBetweenWords: [3000]
})

// 加载单词数据
  onMounted(async () => {
    // 检查用户偏好的主题
    const savedTheme = localStorage.getItem('theme')
    if (savedTheme) {
      isDarkMode.value = savedTheme === 'dark'
    } else {
      isDarkMode.value = window.matchMedia('(prefers-color-scheme: dark)').matches
    }
    updateTheme()

    try {
      const response = await fetch('/8-1.json')
      const data = await response.json()
      
      // 为每个单词生成唯一id
      Object.keys(data).forEach(unit => {
        data[unit].forEach((word, index) => {
          // 使用单元名和索引生成唯一id
          word.uniqueId = `${unit}_${word.id}`
        })
      })
      
      wordData.value = data
    } catch (error) {
      console.error('加载单词数据失败:', error)
    }
  })

// 更新主题
const updateTheme = () => {
  if (isDarkMode.value) {
    document.documentElement.classList.add('dark')
    localStorage.setItem('theme', 'dark')
  } else {
    document.documentElement.classList.remove('dark')
    localStorage.setItem('theme', 'light')
  }
}

// 切换主题
const toggleTheme = () => {
  isDarkMode.value = !isDarkMode.value
  updateTheme()
}

// 开始听写确认
const handleStartDictation = (words) => {
  selectedWords.value = words
  isConfirmActive.value = true
}

// 确认听写
const handleConfirmDictation = (settings) => {
  dictationSettings.value = settings
  isConfirmActive.value = false
  isDictationActive.value = true
}

// 取消确认
const handleCancelConfirm = () => {
  isConfirmActive.value = false
}

// 完成听写返回
const handleFinishDictation = () => {
  isDictationActive.value = false
  isConfirmActive.value = true
}

// 移除单词
const handleRemoveWord = (wordId) => {
  selectedWords.value = selectedWords.value.filter(word => word.uniqueId !== wordId)
}
</script>

<template>
  <div :class="['min-h-screen', isDarkMode ? 'bg-neutral-900 text-neutral-100' : 'bg-neutral-50 text-neutral-900']">
    <!-- Header -->
    <header class="sticky top-0 z-50 backdrop-blur-md" :class="isDarkMode ? 'bg-neutral-900/90 border-b border-neutral-800' : 'bg-neutral-50/90 border-b border-neutral-200'">
      <div class="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <div class="w-10 h-10 bg-primary-500 rounded-lg flex items-center justify-center text-white font-bold text-xl">
            WW
          </div>
          <div>
            <h1 class="text-xl font-bold">英语单词听写</h1>
            <p class="text-xs text-muted-foreground">你的单词听写小助手</p>
          </div>
        </div>
        <Button variant="ghost" size="icon" @click="toggleTheme">
          {{ isDarkMode ? '☀️' : '🌙' }}
        </Button>
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 py-8">
      <div v-if="!isDictationActive && !isConfirmActive && Object.keys(wordData).length > 0">
        <WordList 
          :word-data="wordData" 
          v-model:selected-words="selectedWords"
          @start-dictation="handleStartDictation"
        />
      </div>
      <div v-else-if="isConfirmActive">
        <DictationConfirm 
          :words="selectedWords" 
          @confirm="handleConfirmDictation"
          @cancel="handleCancelConfirm"
          @remove-word="handleRemoveWord"
        />
      </div>
      <div v-else-if="isDictationActive">
        <Dictation 
          :words="selectedWords" 
          :settings="dictationSettings"
          @finish="handleFinishDictation"
        />
      </div>
      <div v-else>
        <Card class="max-w-md mx-auto mt-20" :class="isDarkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'">
          <CardContent class="pt-8">
            <div class="text-center">
              <CardTitle class="text-2xl font-bold" :class="isDarkMode ? 'text-white' : 'text-neutral-800'">加载中...</CardTitle>
              <p :class="isDarkMode ? 'text-neutral-400' : 'text-neutral-600'">正在加载单词数据，请稍候...</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </main>

    <!-- Footer -->
    <footer class="mt-auto py-6" :class="isDarkMode ? 'border-t border-neutral-800' : 'border-t border-neutral-200'">
      <div class="max-w-7xl mx-auto px-4 text-center text-sm" :class="isDarkMode ? 'text-neutral-400' : 'text-neutral-600'">
        <p>© 2025 英语单词听写</p>
      </div>
    </footer>
  </div>
</template>
