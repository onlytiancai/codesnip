<script setup>
import { ref, onMounted, watch } from 'vue'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Slider } from '@/components/ui/slider'
import { Checkbox } from '@/components/ui/checkbox'

const props = defineProps({
  words: {
    type: Array,
    required: true
  }
})

const emit = defineEmits(['confirm', 'cancel', 'remove-word'])

// 从localStorage加载设置
const loadSettingsFromLocalStorage = () => {
  try {
    const savedSettings = localStorage.getItem('dictationSettings')
    if (savedSettings) {
      return JSON.parse(savedSettings)
    }
  } catch (error) {
    console.error('Failed to load settings from localStorage:', error)
  }
  return {
    repeatCount: [2],
    pauseBetweenWords: [3000],
    playChinese: true,
    playEnglish: false,
    showEnglish: true,
    showChinese: true
  }
}

// 设置
const settings = ref(loadSettingsFromLocalStorage())

// 监听设置变化，保存到localStorage
watch(settings, (newSettings) => {
  try {
    localStorage.setItem('dictationSettings', JSON.stringify(newSettings))
  } catch (error) {
    console.error('Failed to save settings to localStorage:', error)
  }
}, { deep: true })

// 移除单词
const removeWord = (wordId) => {
  emit('remove-word', wordId)
}

// 确认听写
const handleConfirm = () => {
  emit('confirm', settings.value)
}

// 取消听写
const handleCancel = () => {
  emit('cancel')
}
</script>

<template>
  <div class="container mx-auto p-4">
    <Card class="max-w-4xl mx-auto" :class="{'bg-white dark:bg-neutral-800': true}">
      <CardHeader class="flex flex-row items-center justify-between pb-6">
        <CardTitle class="text-2xl font-bold">确认听写</CardTitle>
      </CardHeader>
      <CardContent>
        <!-- 所选单词 -->
        <div class="mb-8">
          <h3 class="text-lg font-semibold mb-4">你已选择 {{ words.length }} 个单词:</h3>
          <div class="flex flex-wrap gap-2">
          <div 
            v-for="(word, index) in words" 
            :key="word.uniqueId"
            class="flex items-center gap-2 px-3 py-1.5 rounded-full text-sm"
            :class="{'bg-primary-50 dark:bg-primary-900 text-primary-700 dark:text-primary-200': true}"
          >
            <span>{{ word.word || word.phrase }}</span>
            <Button 
              variant="ghost" 
              size="icon" 
              class="h-5 w-5 p-0 hover:bg-red-100 dark:hover:bg-red-900/30 hover:text-red-600 dark:hover:text-red-400 transition-all"
              @click="removeWord(word.uniqueId)"
            >
              ✕
            </Button>
          </div>
        </div>
        </div>

        <!-- 设置区域 -->
        <div class="bg-neutral-50 dark:bg-neutral-800 p-4 rounded-lg border mb-6">
          <h3 class="text-lg font-semibold mb-4">听写设置</h3>
          
          <!-- 单词朗读次数滑杆 -->
          <div class="mb-4">
            <label class="block text-sm font-medium mb-2">每个单词朗读次数: {{ settings.repeatCount[0] }}</label>
            <Slider 
              v-model="settings.repeatCount" 
              :min="1" 
              :max="5" 
              :step="1"
              class="mb-2"
            />
            <div class="flex justify-between text-xs text-neutral-500 dark:text-neutral-400">
              <span>1次</span>
              <span>2次</span>
              <span>3次</span>
              <span>4次</span>
              <span>5次</span>
            </div>
          </div>

          <!-- 单词停顿滑杆 -->
          <div class="mb-6">
            <label class="block text-sm font-medium mb-2">单词间停顿: {{ settings.pauseBetweenWords[0] / 1000 }}秒</label>
            <Slider 
              v-model="settings.pauseBetweenWords" 
              :min="1000" 
              :max="10000" 
              :step="500"
              class="mb-2"
            />
            <div class="flex justify-between text-xs text-neutral-500 dark:text-neutral-400">
              <span>1秒</span>
              <span>2.5秒</span>
              <span>5秒</span>
              <span>7.5秒</span>
              <span>10秒</span>
            </div>
          </div>

          <!-- 音频播放设置 -->
          <div class="mb-6">
            <h4 class="text-sm font-semibold mb-3">音频播放选项</h4>
            <div class="flex items-center space-x-4 mb-3">
              <Checkbox id="play-chinese" v-model="settings.playChinese" />
              <label for="play-chinese" class="text-sm font-medium">播放中文</label>
            </div>
            <div class="flex items-center space-x-4">
              <Checkbox id="play-english" v-model="settings.playEnglish" />
              <label for="play-english" class="text-sm font-medium">播放英文</label>
            </div>
          </div>

          <!-- 文本显示设置 -->
          <div>
            <h4 class="text-sm font-semibold mb-3">文本显示选项</h4>
            <div class="flex items-center space-x-4 mb-3">
              <Checkbox id="show-english" v-model="settings.showEnglish" />
              <label for="show-english" class="text-sm font-medium">显示英文</label>
            </div>
            <div class="flex items-center space-x-4">
              <Checkbox id="show-chinese" v-model="settings.showChinese" />
              <label for="show-chinese" class="text-sm font-medium">显示中文</label>
            </div>
          </div>
        </div>

        <!-- 控制按钮 -->
        <div class="flex justify-end gap-4">
          <Button variant="secondary" @click="handleCancel" class="transition-all hover:bg-gray-100 dark:hover:bg-gray-800 hover:shadow-md">返回</Button>
          <Button 
            variant="default" 
            @click="handleConfirm"
            :disabled="words.length === 0"
            class="bg-primary-500 hover:bg-primary-600 transition-all hover:shadow-lg"
          >
            开始听写
          </Button>
        </div>
      </CardContent>
    </Card>
  </div>
</template>