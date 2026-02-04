<script setup lang="ts">
import type { WordWithPhoneme } from '../utils/phonemizer';

interface Props {
  words: WordWithPhoneme[];
  sentenceIndex: number;
  showPhonemes: boolean;
  alignPhonemesWithWords: boolean;
  isAiSpeaking?: boolean;
  translation?: string;
  isTranslating?: boolean;
  hasAudio?: boolean;
  showTranslation?: boolean;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  (e: 'speak', sentenceIndex: number): void;
  (e: 'translate', sentenceIndex: number): void;
  (e: 'toggleTranslation', sentenceIndex: number): void;
  (e: 'stop-speaking'): void;
}>();


</script>

<template>
  <div class="sentence-analysis mb-8 p-4 rounded-lg shadow-sm border border-gray-100" :class="{ 'bg-green-50 border-green-200': isAiSpeaking }">
    <div class="flex items-start mb-3">
      <div class="w-6 h-6 rounded-full bg-blue-600 text-white flex items-center justify-center text-sm font-medium mr-3 mt-0.5 flex-shrink-0">
        {{ sentenceIndex + 1 }}
      </div>
      
      <div class="flex-1">
        <!-- 单词和音标组合 - 对齐模式 -->
        <div v-if="showPhonemes && alignPhonemesWithWords" class="flex flex-wrap gap-5">
          <div v-for="(item, index) in words" :key="index" class="text-center min-w-[70px]">
            <div class="font-medium text-gray-900">{{ item.word }}</div>
            <div class="text-sm text-gray-600 mt-1">{{ item.phoneme }}</div>
          </div>
        </div>
        
        <!-- 单词和音标分开显示 - 非对齐模式 -->
        <div v-else-if="showPhonemes && !alignPhonemesWithWords">
          <!-- 单词行 -->
          <div class="flex flex-wrap gap-3 mb-3">
            <span v-for="(item, index) in words" :key="index" class="font-medium text-gray-900">
              {{ item.word }}
              <span v-if="index < words.length - 1"> </span>
            </span>
          </div>
          
          <!-- 音标行 -->
          <div class="flex flex-wrap gap-3 text-sm text-gray-600">
            <span v-for="(item, index) in words" :key="index">
              {{ item.phoneme }}
              <span v-if="index < words.length - 1"> </span>
            </span>
          </div>
        </div>
        
        <!-- 只显示单词 - 不显示音标模式 -->
        <div v-else class="flex flex-wrap gap-3">
          <span v-for="(item, index) in words" :key="index" class="font-medium text-gray-900">
            {{ item.word }}
            <span v-if="index < words.length - 1"> </span>
          </span>
        </div>
      </div>
    </div>
    
    <!-- 翻译结果显示 -->
    <div v-if="props.translation && props.showTranslation" class="mt-3 p-3 bg-blue-50 rounded-md">
      <div class="text-sm font-medium text-gray-700 mb-1">翻译：</div>
      <div class="text-sm text-gray-800">{{ props.translation }}</div>
    </div>
    
    <!-- 操作按钮 -->
    <div class="mt-4 flex flex-wrap gap-3">
      <!-- AI朗读按钮 - 只有生成音频后才显示 -->
      <button 
        v-if="props.hasAudio && !props.isAiSpeaking"
        @click="emit('speak', sentenceIndex)"
        class="px-4 py-2 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors bg-purple-600 text-white hover:bg-purple-700 focus:ring-purple-500"
      >
        播放
      </button>
      
      <!-- 停止朗读按钮 - 只有AI朗读时才显示 -->
      <button 
        v-if="props.hasAudio && props.isAiSpeaking"
        @click="emit('stop-speaking')"
        class="px-4 py-2 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors bg-red-600 text-white hover:bg-red-700 focus:ring-red-500"
      >
        停止
      </button>
      
      <!-- 翻译/显示翻译按钮 -->
      <button 
        @click="props.translation ? emit('toggleTranslation', sentenceIndex) : emit('translate', sentenceIndex)"
        :disabled="props.isTranslating"
        :class="[
          'px-4 py-2 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors',
          props.isTranslating 
            ? 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500 opacity-50 cursor-not-allowed' 
            : props.translation 
              ? (props.showTranslation ? 'bg-gray-600 text-white hover:bg-gray-700 focus:ring-gray-500' : 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500') 
              : 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500'
        ]"
      >
        {{ props.isTranslating ? '翻译中...' : (props.translation ? (props.showTranslation ? '隐藏翻译' : '显示翻译') : '显示翻译') }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.sentence-analysis {
  transition: all 0.2s ease;
}

.sentence-analysis:hover {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}
</style>
