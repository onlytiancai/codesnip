<script setup lang="ts">
import { ref } from 'vue';
import { phonemize } from 'phonemizer';

const inputText = ref('');
const analysisResults = ref<Array<{ words: string[], phonemes: string[] }>>([]);
const isLoading = ref(false);

async function analyzeText() {
  if (!inputText.value.trim()) return;
  
  isLoading.value = true;
  try {
    // 按句子拆分文本
    const sentences = inputText.value.split(/[.!?]+/).filter(s => s.trim());
    const results: Array<{ words: string[], phonemes: string[] }> = [];
    
    for (const sentence of sentences) {
      const words = sentence.trim().split(/\s+/).filter(w => w);
      if (words.length === 0) continue;
      
      // 为每个单词生成音标
      const phonemes: string[] = [];
      for (const word of words) {
        const phoneme = await phonemize(word);
        if (phoneme[0]) {
          phonemes.push(phoneme[0]);
        } else {
          phonemes.push('');
        }
      }
      
      results.push({ words, phonemes });
    }
    
    analysisResults.value = results;
  } catch (error) {
    console.error('分析文本时出错:', error);
  } finally {
    isLoading.value = false;
  }
}
</script>

<template>
  <div class="container mx-auto px-4 py-8 max-w-3xl">
    <h1 class="text-3xl font-bold mb-6 text-center">英文音标分析器</h1>
    
    <div class="mb-6">
      <label for="text-input" class="block text-sm font-medium text-gray-700 mb-2">输入英文文本：</label>
      <textarea 
        id="text-input"
        v-model="inputText"
        class="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        rows="4"
        placeholder="请输入英文文本，例如：Hello world. How are you?"
      ></textarea>
    </div>
    
    <div class="mb-8">
      <button 
        @click="analyzeText"
        :disabled="isLoading"
        class="px-6 py-2 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed"
      >
        {{ isLoading ? '分析中...' : '分析文本' }}
      </button>
    </div>
    
    <div v-if="analysisResults.length > 0" class="mt-8">
      <h2 class="text-xl font-semibold mb-4">分析结果：</h2>
      
      <div v-for="(result, index) in analysisResults" :key="index" class="mb-8">
        <div class="mb-2 font-medium">句子 {{ index + 1 }}：</div>
        
        <!-- 单词和音标组合 -->
        <div class="flex flex-wrap gap-4">
          <div v-for="(word, index) in result.words" :key="index" class="text-center min-w-[60px]">
            <div class="font-medium">{{ word }}</div>
            <div class="text-sm text-gray-600">{{ result.phonemes[index] }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 可以添加自定义样式 */
</style>
