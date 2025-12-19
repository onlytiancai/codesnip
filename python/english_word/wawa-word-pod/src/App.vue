<script setup>
import { ref, onMounted } from 'vue'
import WordList from './components/WordList.vue'
import Dictation from './components/Dictation.vue'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

const wordData = ref({})
const selectedWords = ref([])
const isDictationActive = ref(false)

// 加载单词数据
onMounted(async () => {
  try {
    const response = await fetch('/8-1.json')
    const data = await response.json()
    wordData.value = data
  } catch (error) {
    console.error('加载单词数据失败:', error)
  }
})

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
  <div class="min-h-screen bg-neutral-50">
    <main class="max-w-7xl mx-auto px-4 py-8">
      <div v-if="!isDictationActive && Object.keys(wordData).length > 0">
        <WordList 
          :word-data="wordData" 
          @start-dictation="handleStartDictation"
        />
      </div>
      <div v-else-if="isDictationActive">
        <Dictation 
          :words="selectedWords" 
          @finish="handleFinishDictation"
        />
      </div>
      <div v-else>
        <Card class="max-w-md mx-auto mt-20">
          <CardContent class="pt-8">
            <div class="text-center">
              <CardTitle class="text-2xl font-bold text-neutral-800 mb-2">加载中...</CardTitle>
              <p class="text-neutral-600">正在加载单词数据，请稍候...</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  </div>
</template>
