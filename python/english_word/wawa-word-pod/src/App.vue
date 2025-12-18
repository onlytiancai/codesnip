<script setup>
import { ref, onMounted } from 'vue'
import WordList from './components/WordList.vue'
import Dictation from './components/Dictation.vue'

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
  <div class="app">
    <main>
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
        <div class="loading">
          <h2>加载中...</h2>
          <p>正在加载单词数据，请稍候...</p>
        </div>
      </div>
    </main>
  </div>
</template>

<style scoped>
.app {
  min-height: 100vh;
  background-color: #f5f5f5;
}

main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.loading {
  text-align: center;
  padding: 60px 20px;
  color: #666;
}
</style>
