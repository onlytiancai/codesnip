<script setup>
import { ref, onMounted } from 'vue'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'

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
  <div class="container mx-auto p-4">
    <Card class="max-w-4xl mx-auto">
      <CardHeader class="flex flex-row items-center justify-between pb-6">
        <CardTitle class="text-2xl font-bold text-neutral-800">英语单词听写</CardTitle>
        <div class="flex gap-2">
          <Button variant="secondary" @click="selectAllWords">全选</Button>
          <Button variant="secondary" @click="clearSelection">清空</Button>
          <Button 
            variant="default" 
            @click="handleStartDictation"
            :disabled="selectedWords.length === 0"
          >
            开始听写 ({{ selectedWords.length }})
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div class="space-y-3">
          <div 
            v-for="(items, unit) in wordData" 
            :key="unit"
            class="border rounded-lg overflow-hidden"
          >
            <div 
              class="flex items-center justify-between p-4 bg-neutral-50 cursor-pointer hover:bg-neutral-100 transition-colors"
              @click="toggleUnit(unit)"
            >
              <h3 class="font-semibold text-neutral-800">{{ unit }}</h3>
              <span class="text-neutral-500">{{ isUnitExpanded(unit) ? '▼' : '▶' }}</span>
            </div>
            
            <div v-if="isUnitExpanded(unit)" class="p-3">
              <div 
                v-for="item in items" 
                :key="item.id"
                class="flex items-center justify-between p-3 rounded-md cursor-pointer hover:bg-neutral-50 transition-colors"
                :class="{ 'bg-green-50 border-l-4 border-green-500': isWordSelected(item) }"
                @click="toggleWordSelection(item)"
              >
                <div class="flex-1">
                  <div class="font-medium text-neutral-800">{{ item.word || item.phrase }}</div>
                  <div class="text-sm text-neutral-600 mt-1">{{ item.chinese }}</div>
                </div>
                <div class="ml-4">
                  <Checkbox 
                    :checked="isWordSelected(item)"
                    @change="toggleWordSelection(item)"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  </div>
</template>

