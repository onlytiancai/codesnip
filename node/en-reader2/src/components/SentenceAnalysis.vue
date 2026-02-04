<script setup lang="ts">
import type { WordWithPhoneme } from '../utils/phonemizer';

interface Props {
  words: WordWithPhoneme[];
  sentenceIndex: number;
}

const props = defineProps<Props>();
</script>

<template>
  <div class="sentence-analysis mb-8 bg-white p-4 rounded-lg shadow-sm border border-gray-100">
    <div class="mb-3 font-medium text-gray-700">句子 {{ sentenceIndex + 1 }}：</div>
    
    <!-- 单词和音标组合 -->
    <div class="flex flex-wrap gap-5">
      <div v-for="(item, index) in words" :key="index" class="text-center min-w-[70px]">
        <div class="font-medium text-gray-900">{{ item.word }}</div>
        <div class="text-sm text-gray-600 mt-1">{{ item.phoneme }}</div>
      </div>
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
