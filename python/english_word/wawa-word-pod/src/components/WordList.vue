<script setup>
import { ref, onMounted, watch } from 'vue'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { 
  CheckSquare, Square, ChevronDown, ChevronRight, Trash2, 
  PlayCircle, ListChecks, XCircle 
} from 'lucide-vue-next'

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
const MAX_SELECTION = 100

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

// 全选当前单元的所有单词
const selectAllInUnit = (unit) => {
  const unitWords = props.wordData[unit]
  // 检查是否会超过选择上限
  const selectedCount = localSelectedWords.value.length
  const availableSlots = MAX_SELECTION - selectedCount
  const unitWordsCount = unitWords.length
  
  if (availableSlots < unitWordsCount) {
    alert(`选择该单元会超过 ${MAX_SELECTION} 个单词的限制，最多只能再选择 ${availableSlots} 个单词。`)
    return
  }
  
  // 添加该单元的所有单词
  unitWords.forEach(wordItem => {
    if (!isWordSelected(wordItem)) {
      localSelectedWords.value.push(wordItem)
    }
  })
  emit('update:selectedWords', localSelectedWords.value)
}

// 清空当前单元的所有已选单词
const clearAllInUnit = (unit) => {
  const unitWords = props.wordData[unit]
  // 过滤掉该单元的所有已选单词
  localSelectedWords.value = localSelectedWords.value.filter(item => {
    return !unitWords.some(wordItem => wordItem.uniqueId === item.uniqueId)
  })
  emit('update:selectedWords', localSelectedWords.value)
}

const handleStartDictation = () => {
  if (localSelectedWords.value.length > 0) {
    emit('start-dictation', localSelectedWords.value)
  }
}
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 p-4 md:p-6">
    <div class="max-w-6xl mx-auto">
      <!-- 顶部卡片 -->
      <Card class="mb-6 overflow-hidden shadow-lg hover:shadow-xl transition-shadow duration-300 border-0 bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm">
        <CardHeader class="bg-gradient-to-r from-primary-500 to-primary-600 text-white">
          <div class="flex items-center gap-3">
            <ListChecks class="h-8 w-8" />
            <CardTitle class="text-2xl md:text-3xl font-bold">单词列表</CardTitle>
          </div>
  
        </CardHeader>
        <CardContent>
          <div class="mt-4 mb-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-blue-700 dark:text-blue-300 text-sm flex items-start gap-2">
            <div class="flex-shrink-0 mt-0.5">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-blue-500">
                <circle cx="12" cy="12" r="10" />
                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                <path d="M12 17h.01" />
              </svg>
            </div>
            <div>
              <span class="font-medium">提示：</span>点击单词选择要听写的单词，然后点击底部的"开始听写"按钮
            </div>
          </div>
          
          <!-- 选择统计 -->
          <div class="mb-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg text-green-700 dark:text-green-300 text-sm flex items-center justify-between">
            <div class="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-green-500">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
              <span>已选择 <strong>{{ localSelectedWords.length }}</strong> / {{ MAX_SELECTION }} 个单词</span>
            </div>

                    <div class="flex gap-2 flex-wrap mt-2">

          </div>

            <div class="flex items-center gap-2">
              <Button 
                variant="secondary" 
                @click="clearSelection" 
                class="transition-all hover:bg-red-100 hover:text-red-600 dark:hover:bg-red-900/30 dark:hover:text-red-400 hover:shadow-md"
              >
                <Trash2 class="h-3.5 w-3.5 mr-1" />
                清空
              </Button>
            </div>
          </div>
          
          <!-- 单元列表 -->
          <div class="space-y-4">
            <div 
              v-for="(items, unit) in wordData" 
              :key="unit"
              class="border rounded-xl overflow-hidden shadow-md hover:shadow-lg transition-all duration-300 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm"
            >
              <div 
                class="flex items-center justify-between p-4 cursor-pointer transition-all duration-200"
                :class="{ 'hover:bg-slate-100 dark:hover:bg-slate-700/70': true }"
                @click="toggleUnit(unit)"
              >
                <div class="flex items-center gap-3">
                  <div class="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center text-primary-600 dark:text-primary-400 font-semibold">
                    {{ unit.replace(/[^0-9]/g, '') }}
                  </div>
                  <h3 class="font-semibold text-lg text-slate-800 dark:text-white">{{ unit }}</h3>
                  <span class="text-xs px-2 py-1 bg-slate-100 dark:bg-slate-700 rounded-full text-slate-600 dark:text-slate-300">
                    {{ items.length }} 个单词
                  </span>
                </div>
                <div class="flex items-center gap-3">
                  <div class="flex gap-1">
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      class="h-8 w-8 p-0 hover:bg-green-100 dark:hover:bg-green-900/30 hover:text-green-600 dark:hover:text-green-400 transition-colors"
                      @click.stop="selectAllInUnit(unit)"
                      title="全选本单元"
                    >
                      <CheckSquare class="h-4 w-4" />
                    </Button>
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      class="h-8 w-8 p-0 hover:bg-red-100 dark:hover:bg-red-900/30 hover:text-red-600 dark:hover:text-red-400 transition-colors"
                      @click.stop="clearAllInUnit(unit)"
                      title="清空本单元"
                    >
                      <XCircle class="h-4 w-4" />
                    </Button>
                  </div>
                  <div class="w-8 h-8 rounded-full bg-slate-100 dark:bg-slate-700 flex items-center justify-center">
                    <ChevronDown 
                      v-if="isUnitExpanded(unit)" 
                      class="h-4 w-4 text-slate-600 dark:text-slate-300 transition-transform duration-300"
                    />
                    <ChevronRight 
                      v-else 
                      class="h-4 w-4 text-slate-600 dark:text-slate-300 transition-transform duration-300"
                    />
                  </div>
                </div>
              </div>
              
              <div 
                v-if="isUnitExpanded(unit)" 
                class="p-4 bg-slate-50 dark:bg-slate-900/50 border-t"
              >
                <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
                  <div 
                    v-for="item in items" 
                    :key="item.id"
                    class="group relative p-4 rounded-xl cursor-pointer transition-all duration-300 transform hover:scale-105"
                    :class="isWordSelected(item) 
                      ? 'bg-primary-500 text-white shadow-lg hover:bg-primary-600' 
                      : 'bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700'"
                    @click="toggleWordSelection(item)"
                  >
                    <div class="absolute top-3 right-3">
                      <CheckSquare 
                        v-if="isWordSelected(item)" 
                        class="h-4 w-4 text-white" 
                      />
                      <Square 
                        v-else 
                        class="h-4 w-4 text-slate-400 dark:text-slate-500 group-hover:text-primary-500 transition-colors"
                      />
                    </div>
                    <div class="font-semibold text-lg mb-2">{{ item.word || item.phrase }}</div>
                    <div class="text-sm opacity-90 line-clamp-2">{{ item.chinese }}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <!-- 悬浮的开始听写按钮 -->
      <div class="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-50">
        <Button 
          variant="default" 
          @click="handleStartDictation"
          :disabled="localSelectedWords.length === 0"
          class="group flex items-center gap-2 px-8 py-6 text-lg font-semibold shadow-xl rounded-full transition-all duration-300"
          :class="localSelectedWords.length > 0 
            ? 'bg-primary-500 hover:bg-primary-600 hover:shadow-2xl hover:scale-105' 
            : 'bg-slate-300 hover:bg-slate-400 text-slate-600 cursor-not-allowed'"
        >
          <PlayCircle class="h-6 w-6 group-hover:scale-110 transition-transform" />
          <span>开始听写</span>
          <span class="bg-white/20 px-3 py-1 rounded-full text-sm backdrop-blur-sm">
            {{ localSelectedWords.length }}
          </span>
        </Button>
      </div>
    </div>
  </div>
</template>

