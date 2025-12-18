<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'

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
  <div class="dictation">
    <div class="header">
      <h2>听写界面</h2>
      <button @click="goBack" class="btn btn-secondary">返回</button>
    </div>

    <!-- 设置区域 -->
    <div class="settings" v-if="!isPlaying && !dictationComplete">
      <h3>听写设置</h3>
      <div class="setting-item">
        <label for="repeatCount">每个单词读取次数:</label>
        <input 
          id="repeatCount" 
          type="number" 
          v-model.number="settings.repeatCount" 
          min="1" 
          max="5"
        />
      </div>
      <div class="setting-item">
        <label for="pauseBetweenWords">单词间停顿(毫秒):</label>
        <input 
          id="pauseBetweenWords" 
          type="number" 
          v-model.number="settings.pauseBetweenWords" 
          min="1000" 
          max="10000"
          step="500"
        />
      </div>
    </div>

    <!-- 听写进度 -->
    <div class="progress" v-if="!dictationComplete">
      <div class="progress-info">
        <div>当前单词: {{ currentWordIndex + 1 }} / {{ words.length }}</div>
        <div v-if="currentWordIndex < words.length">
          当前重复: {{ currentRepeat + 1 }} / {{ settings.repeatCount }}
        </div>
      </div>
      <div class="progress-bar">
        <div 
          class="progress-fill" 
          :style="{ width: `${((currentWordIndex + (currentRepeat / settings.repeatCount)) / words.length) * 100}%` }"
        ></div>
      </div>
    </div>

    <!-- 听写内容 -->
    <div class="dictation-content">
      <div v-if="dictationComplete" class="complete-message">
        <h3>恭喜您完成听写！</h3>
        <p>您总共听写了 {{ words.length }} 个单词/短语。</p>
      </div>
      <div v-else-if="currentWordIndex < words.length" class="current-word">
        <div class="word-info">
          <div class="word-text">
            {{ words[currentWordIndex].word || words[currentWordIndex].phrase }}
          </div>
          <div class="word-chinese">
            {{ words[currentWordIndex].chinese }}
          </div>
        </div>
      </div>
    </div>

    <!-- 控制按钮 -->
    <div class="controls">
      <div v-if="!dictationComplete">
        <button 
          @click="!isPlaying ? startDictation() : (isPaused ? resumeDictation() : pauseDictation())"
          class="btn btn-primary"
        >
          {{ !isPlaying ? '开始听写' : (isPaused ? '继续' : '暂停') }}
        </button>
        <button 
          @click="restartDictation"
          class="btn btn-secondary"
        >
          重新开始
        </button>
      </div>
      <div v-else>
        <button 
          @click="restartDictation"
          class="btn btn-primary"
        >
          再次听写
        </button>
        <button 
          @click="goBack"
          class="btn btn-secondary"
        >
          返回单词列表
        </button>
      </div>
    </div>

    <!-- 隐藏的音频元素 -->
    <audio 
      ref="audioElement"
      @ended="handleAudioEnd"
      preload="auto"
    ></audio>
  </div>
</template>

<style scoped>
.dictation {
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid #e0e0e0;
}

.header h2 {
  margin: 0;
  color: #333;
}

.settings {
  margin: 20px 0;
  padding: 16px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background-color: #f8f9fa;
}

.settings h3 {
  margin-top: 0;
  margin-bottom: 16px;
  color: #333;
}

.setting-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.setting-item:last-child {
  margin-bottom: 0;
}

.setting-item input {
  width: 100px;
  padding: 6px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.progress {
  margin: 20px 0;
}

.progress-info {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 14px;
  color: #666;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background-color: #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background-color: #42b983;
  transition: width 0.3s ease;
}

.dictation-content {
  text-align: center;
  margin: 40px 0;
}

.complete-message {
  padding: 30px;
  background-color: #d4edda;
  border: 1px solid #c3e6cb;
  border-radius: 8px;
  color: #155724;
}

.complete-message h3 {
  margin-top: 0;
}

.current-word {
  padding: 30px;
  background-color: #f8f9fa;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
}

.word-info {
  display: inline-block;
  text-align: left;
}

.word-text {
  font-size: 24px;
  font-weight: bold;
  color: #333;
  margin-bottom: 8px;
}

.word-chinese {
  font-size: 18px;
  color: #666;
}

.controls {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 30px;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s;
}

.btn-primary {
  background-color: #42b983;
  color: white;
}

.btn-primary:hover {
  background-color: #359f70;
}

.btn-secondary {
  background-color: #f0f0f0;
  color: #333;
}

.btn-secondary:hover {
  background-color: #e0e0e0;
}
</style>