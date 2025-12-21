<script setup>
import { ref, onMounted, computed } from 'vue'
import WordList from './components/WordList.vue'
import Dictation from './components/Dictation.vue'
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

// 课本切换相关状态
const currentTextbook = ref({
  id: 'lu-jiao-ban-old-8a',
  name: '鲁教版（旧版）八年级上册',
  file: '8-1.json'
})

const textbooks = ref([
  {
    id: 'lu-jiao-ban-old-8a',
    name: '鲁教版（旧版）八年级上册',
    file: '8-1.json',
    available: true
  },
  {
    id: 'lu-jiao-ban-old-8b',
    name: '鲁教版（旧版）八年级下册',
    file: '8-2.json',
    available: false
  },
  {
    id: 'lu-jiao-ban-old-9a',
    name: '鲁教版（旧版）九年级上册',
    file: '9-1.json',
    available: false
  },
  {
    id: 'lu-jiao-ban-old-9b',
    name: '鲁教版（旧版）九年级下册',
    file: '9-2.json',
    available: false
  }
])

// 切换课本
const switchTextbook = async (textbook) => {
  if (!textbook.available) {
    alert('该课本正在开发中，敬请期待！')
    return
  }
  
  // 清除当前选择的单词
  selectedWords.value = []
  
  // 更新当前课本
  currentTextbook.value = textbook
  
  // 加载新课本的单词数据
  await loadWordData(textbook.file)
}

// 加载单词数据
const loadWordData = async (file) => {
  try {
    const response = await fetch(`/${file}`)
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
}

onMounted(async () => {
  // 检查用户偏好的主题
  const savedTheme = localStorage.getItem('theme')
  if (savedTheme) {
    isDarkMode.value = savedTheme === 'dark'
  } else {
    isDarkMode.value = window.matchMedia('(prefers-color-scheme: dark)').matches
  }
  updateTheme()

  // 加载当前课本的单词数据
  await loadWordData(currentTextbook.value.file)
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

// 开始听写
const handleStartDictation = (words) => {
  selectedWords.value = words
  isDictationActive.value = true
}

// 完成听写返回
const handleFinishDictation = () => {
  isDictationActive.value = false
}


</script>

<template>
  <div class="max-w-2xl mx-auto" :class="['min-h-screen', isDarkMode ? 'bg-neutral-900 text-neutral-100' : 'bg-neutral-50 text-neutral-900']">



        <div v-if="!isDictationActive && !isConfirmActive && Object.keys(wordData).length > 0">
        <WordList 
          :word-data="wordData" 
          v-model:selected-words="selectedWords"
          @start-dictation="handleStartDictation"
          :current-textbook="currentTextbook"
          :textbooks="textbooks"
          :is-dark-mode="isDarkMode"
          :on-toggle-theme="toggleTheme"
          @switch-textbook="switchTextbook"
        />
      </div>

      <div v-else-if="isDictationActive">
        <Dictation 
          :words="selectedWords" 
          :settings="dictationSettings"
          :is-dark-mode="isDarkMode"
          :on-toggle-theme="toggleTheme"
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
    <!-- Footer -->
    <footer class="mt-auto py-6" :class="isDarkMode ? 'border-t border-neutral-800' : 'border-t border-neutral-200'">
      <div class="max-w-7xl mx-auto px-4 text-center text-sm" :class="isDarkMode ? 'text-neutral-400' : 'text-neutral-600'">
        <p>© 2025 英语单词听写</p>
      </div>
    </footer>
  </div>
</template>
