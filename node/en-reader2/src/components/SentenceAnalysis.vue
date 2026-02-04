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
}

const props = defineProps<Props>();

const emit = defineEmits<{
  (e: 'speak', sentenceIndex: number): void;
  (e: 'translate', sentenceIndex: number): void;
}>();

// 获取完整句子
const getFullSentence = (): string => {
  return props.words.map(word => word.word).join(' ');
};
</script>

<template>
  <div class="sentence-analysis mb-8 bg-white p-4 rounded-lg shadow-sm border border-gray-100">
    <div class="mb-3 font-medium text-gray-700">句子 {{ sentenceIndex + 1 }}：</div>
    
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
    
    <!-- 翻译结果显示 -->
    <div v-if="props.translation" class="mt-3 p-3 bg-blue-50 rounded-md">
      <div class="text-sm font-medium text-gray-700 mb-1">翻译：</div>
      <div class="text-sm text-gray-800">{{ props.translation }}</div>
    </div>
    
    <!-- 操作按钮 -->
    <div class="mt-4 flex gap-3">
      <!-- AI朗读按钮 -->
      <button 
        @click="emit('speak', sentenceIndex)"
        :disabled="props.isAiSpeaking"
        :class="[
          'px-4 py-2 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors',
          props.isAiSpeaking 
            ? 'bg-purple-600 text-white hover:bg-purple-700 focus:ring-purple-500 opacity-50 cursor-not-allowed' 
            : 'bg-purple-600 text-white hover:bg-purple-700 focus:ring-purple-500'
        ]"
      >
        {{ props.isAiSpeaking ? 'AI朗读中...' : 'AI朗读' }}
      </button>
      
      <!-- 翻译按钮 -->
      <button 
        @click="emit('translate', sentenceIndex)"
        :disabled="props.isTranslating"
        :class="[
          'px-4 py-2 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors',
          props.isTranslating 
            ? 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500 opacity-50 cursor-not-allowed' 
            : 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500'
        ]"
      >
        {{ props.isTranslating ? '翻译中...' : '翻译' }}
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
