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
    showChinese: true,
    showPhonetic: false,
    shuffle: false
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
          <h3 class="text-base font-semibold mb-3">听写设置</h3>
          
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- 左侧设置 -->
            <div>
              <!-- 单词朗读次数 -->
              <div class="mb-3">
                <label class="block text-xs font-medium mb-1">播放次数</label>
                <select 
                  v-model="settings.repeatCount[0]" 
                  class="w-full px-2 py-1.5 text-sm border rounded-md bg-white dark:bg-neutral-700 dark:border-neutral-600 dark:text-white"
                >
                  <option :value="1">1次</option>
                  <option :value="2">2次</option>
                  <option :value="3">3次</option>
                  <option :value="4">4次</option>
                  <option :value="5">5次</option>
                </select>
              </div>

              <!-- 单词停顿间隔 -->
              <div class="mb-3">
                <label class="block text-xs font-medium mb-1">播放间隔</label>
                <select 
                  v-model="settings.pauseBetweenWords[0]" 
                  class="w-full px-2 py-1.5 text-sm border rounded-md bg-white dark:bg-neutral-700 dark:border-neutral-600 dark:text-white"
                >
                  <option :value="1000">1秒</option>
                  <option :value="2000">2秒</option>
                  <option :value="3000">3秒</option>
                  <option :value="4000">4秒</option>
                  <option :value="5000">5秒</option>
                  <option :value="6000">6秒</option>
                  <option :value="7000">7秒</option>
                  <option :value="8000">8秒</option>
                  <option :value="9000">9秒</option>
                  <option :value="10000">10秒</option>
                </select>
              </div>
            </div>

            <!-- 右侧设置 -->
            <div>
              <!-- 音频播放设置 -->
              <div class="mb-3">
                <h4 class="text-xs font-semibold mb-2">音频播放选项</h4>
                <div class="flex items-center space-x-3 mb-2">
                  <Checkbox id="play-chinese" v-model="settings.playChinese" />
                  <label for="play-chinese" class="text-xs font-medium">播放中文</label>
                </div>
                <div class="flex items-center space-x-3 mb-2">
                  <Checkbox id="play-english" v-model="settings.playEnglish" />
                  <label for="play-english" class="text-xs font-medium">播放英文</label>
                </div>
                <div class="flex items-center space-x-3">
                  <Checkbox id="shuffle" v-model="settings.shuffle" />
                  <label for="shuffle" class="text-xs font-medium">随机播放</label>
                </div>
              </div>

              <!-- 文本显示设置 -->
              <div>
                <h4 class="text-xs font-semibold mb-2">文本显示选项</h4>
                <div class="flex items-center space-x-3 mb-2">
                  <Checkbox id="show-english" v-model="settings.showEnglish" />
                  <label for="show-english" class="text-xs font-medium">显示英文</label>
                </div>
                <div class="flex items-center space-x-3 mb-2">
                  <Checkbox id="show-chinese" v-model="settings.showChinese" />
                  <label for="show-chinese" class="text-xs font-medium">显示中文</label>
                </div>
                <div class="flex items-center space-x-3">
                  <Checkbox id="show-phonetic" v-model="settings.showPhonetic" />
                  <label for="show-phonetic" class="text-xs font-medium">显示音标</label>
                </div>
              </div>
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