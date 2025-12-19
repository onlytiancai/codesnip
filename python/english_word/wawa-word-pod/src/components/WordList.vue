<script setup>
import { ref, onMounted, watch } from 'vue'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'

const props = defineProps({
  wordData: {
    type: Object,
    required: true
  },
  selectedWords: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['select-word', 'start-dictation', 'update:selectedWords'])

const localSelectedWords = ref([])
const expandedUnits = ref([])

onMounted(() => {
  // 默认展开所有单元
  expandedUnits.value = Object.keys(props.wordData)
  // 使用父组件传递的选中状态
  if (props.selectedWords.length > 0) {
    localSelectedWords.value = [...props.selectedWords]
  }
})

// 最大选择单词数
const MAX_SELECTION = 20

// 监听父组件选中状态变化
watch(() => props.selectedWords, (newVal) => {
  if (newVal.length > 0) {
    localSelectedWords.value = [...newVal]
  }
}, { deep: true })

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
  const index = localSelectedWords.value.findIndex(item => item.uniqueId === wordItem.uniqueId)
  if (index > -1) {
    // 取消选择
    localSelectedWords.value.splice(index, 1)
    emit('update:selectedWords', localSelectedWords.value)
  } else {
    // 检查是否达到选择上限
    if (localSelectedWords.value.length >= MAX_SELECTION) {
      alert(`最多只能选择 ${MAX_SELECTION} 个单词，请先取消一些选择。`)
      return
    }
    // 添加选择
    localSelectedWords.value.push(wordItem)
    emit('update:selectedWords', localSelectedWords.value)
  }
}

const isWordSelected = (wordItem) => {
  return localSelectedWords.value.some(item => item.uniqueId === wordItem.uniqueId)
}

const selectAllWords = () => {
  const allWords = []
  Object.values(props.wordData).forEach(items => {
    items.forEach(item => {
      allWords.push(item)
    })
  })
  
  if (allWords.length > MAX_SELECTION) {
    alert(`最多只能选择 ${MAX_SELECTION} 个单词，所有单词数量超过了这个限制。`)
    return
  }
  
  localSelectedWords.value = [...allWords]
  emit('update:selectedWords', localSelectedWords.value)
}

const clearSelection = () => {
  localSelectedWords.value = []
  emit('update:selectedWords', localSelectedWords.value)
}

const handleStartDictation = () => {
  if (localSelectedWords.value.length > 0) {
    emit('start-dictation', localSelectedWords.value)
  }
}
</script>

<template>
  <div class="container mx-auto p-4">
    <Card class="max-w-6xl mx-auto" :class="{'bg-white dark:bg-neutral-800': true}">
      <CardHeader class="flex flex-row items-center justify-between pb-6">
        <CardTitle class="text-2xl font-bold">单词列表</CardTitle>
        <div class="flex gap-2 flex-wrap">
          <Button variant="secondary" @click="selectAllWords">全选</Button>
          <Button variant="secondary" @click="clearSelection">清空</Button>
        </div>
      </CardHeader>
      <CardContent>
        <div class="space-y-4">
          <div 
            v-for="(items, unit) in wordData" 
            :key="unit"
            class="border rounded-lg overflow-hidden"
            :class="{'bg-white dark:bg-neutral-800': true}"
          >
            <div 
              class="flex items-center justify-between p-4 cursor-pointer transition-colors"
              :class="{ 'hover:bg-neutral-100 dark:hover:bg-neutral-700': true }"
              @click="toggleUnit(unit)"
            >
              <h3 class="font-semibold">{{ unit }}</h3>
              <span :class="{ 'text-neutral-500 dark:text-neutral-400': true }">
                {{ isUnitExpanded(unit) ? '▼' : '▶' }}
              </span>
            </div>
            
            <div v-if="isUnitExpanded(unit)" class="p-3">
              <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
                <div 
                  v-for="item in items" 
                  :key="item.id"
                  class="p-3 rounded-lg cursor-pointer transition-all duration-200 transform hover:scale-105"
                  :class="isWordSelected(item) 
                    ? 'bg-primary-500 text-white shadow-lg hover:bg-primary-600' 
                    : 'bg-neutral-100 dark:bg-neutral-800 hover:bg-neutral-200 dark:hover:bg-neutral-700 text-neutral-600 dark:text-neutral-300'"
                  @click="toggleWordSelection(item)"
                >
                  <div class="font-medium mb-1">{{ item.word || item.phrase }}</div>
                  <div class="text-xs opacity-80">{{ item.chinese }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>

    <!-- 悬浮的开始听写按钮 -->
    <div class="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-40">
      <Button 
          variant="default" 
          @click="handleStartDictation"
          :disabled="localSelectedWords.length === 0"
          class="px-8 py-6 text-lg font-semibold shadow-lg"
          :class="localSelectedWords.length > 0 ? 'bg-primary-500 hover:bg-primary-600' : 'bg-neutral-300 hover:bg-neutral-400 text-neutral-600'"
        >
        开始听写 ({{ localSelectedWords.length }})
      </Button>
    </div>
  </div>
</template>

