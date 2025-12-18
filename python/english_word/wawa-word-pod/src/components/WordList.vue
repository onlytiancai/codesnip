<script setup>
import { ref, onMounted } from 'vue'

const props = defineProps({
  wordData: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['select-word', 'start-dictation'])

const selectedWords = ref([])
const expandedUnits = ref([])

onMounted(() => {
  // 默认展开所有单元
  expandedUnits.value = Object.keys(props.wordData)
})

const toggleUnit = (unit) => {
  const index = expandedUnits.value.indexOf(unit)
  if (index > -1) {
    expandedUnits.value.splice(index, 1)
  } else {
    expandedUnits.value.push(unit)
  }
}

const isUnitExpanded = (unit) => {
  return expandedUnits.value.includes(unit)
}

const toggleWordSelection = (wordItem) => {
  const index = selectedWords.value.findIndex(item => item.id === wordItem.id)
  if (index > -1) {
    selectedWords.value.splice(index, 1)
  } else {
    selectedWords.value.push(wordItem)
  }
}

const isWordSelected = (wordItem) => {
  return selectedWords.value.some(item => item.id === wordItem.id)
}

const selectAllWords = () => {
  selectedWords.value = []
  Object.values(props.wordData).forEach(items => {
    items.forEach(item => {
      selectedWords.value.push(item)
    })
  })
}

const clearSelection = () => {
  selectedWords.value = []
}

const handleStartDictation = () => {
  if (selectedWords.value.length > 0) {
    emit('start-dictation', selectedWords.value)
  }
}
</script>

<template>
  <div class="word-list">
    <div class="header">
      <h2>英语单词听写</h2>
      <div class="selection-controls">
        <button @click="selectAllWords" class="btn btn-secondary">全选</button>
        <button @click="clearSelection" class="btn btn-secondary">清空</button>
        <button 
          @click="handleStartDictation" 
          class="btn btn-primary"
          :disabled="selectedWords.length === 0"
        >
          开始听写 ({{ selectedWords.length }})
        </button>
      </div>
    </div>

    <div class="units">
      <div 
        v-for="(items, unit) in wordData" 
        :key="unit"
        class="unit"
      >
        <div class="unit-header" @click="toggleUnit(unit)">
          <h3>{{ unit }}</h3>
          <span class="unit-toggle">{{ isUnitExpanded(unit) ? '▼' : '▶' }}</span>
        </div>
        
        <div v-if="isUnitExpanded(unit)" class="unit-words">
          <div 
            v-for="item in items" 
            :key="item.id"
            class="word-item"
            :class="{ 'selected': isWordSelected(item) }"
            @click="toggleWordSelection(item)"
          >
            <div class="word-content">
              <div class="word-text">
                {{ item.word || item.phrase }}
              </div>
              <div class="word-chinese">
                {{ item.chinese }}
              </div>
            </div>
            <div class="word-select">
              <input 
                type="checkbox"
                :checked="isWordSelected(item)"
                @change="toggleWordSelection(item)"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.word-list {
  max-width: 800px;
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

.selection-controls {
  display: flex;
  gap: 10px;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.3s;
}

.btn-primary {
  background-color: #42b983;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background-color: #359f70;
}

.btn-primary:disabled {
  background-color: #a3d9c0;
  cursor: not-allowed;
}

.btn-secondary {
  background-color: #f0f0f0;
  color: #333;
}

.btn-secondary:hover {
  background-color: #e0e0e0;
}

.units {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.unit {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  overflow: hidden;
}

.unit-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background-color: #f8f9fa;
  cursor: pointer;
  font-weight: bold;
  color: #333;
}

.unit-toggle {
  font-size: 12px;
  color: #666;
}

.unit-words {
  padding: 8px;
}

.word-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 12px;
  margin: 4px 0;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.word-item:hover {
  background-color: #f0f0f0;
}

.word-item.selected {
  background-color: #e8f5e9;
  border-left: 4px solid #42b983;
}

.word-content {
  flex: 1;
}

.word-text {
  font-size: 16px;
  font-weight: 500;
  color: #333;
}

.word-chinese {
  font-size: 14px;
  color: #666;
  margin-top: 2px;
}

.word-select {
  margin-left: 16px;
}

.word-select input {
  width: 18px;
  height: 18px;
}
</style>