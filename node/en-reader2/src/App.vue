<script setup lang="ts">
import { ref } from 'vue';
import { analyzeEnglishText } from './utils/phonemizer';
import type { SentenceAnalysis } from './utils/phonemizer';
import SentenceAnalysisComponent from './components/SentenceAnalysis.vue';

const inputText = ref('');
const analysisResults = ref<SentenceAnalysis[]>([]);
const isLoading = ref(false);
const errorMessage = ref('');

async function analyzeText() {
  if (!inputText.value.trim()) {
    errorMessage.value = '请输入英文文本';
    return;
  }
  
  errorMessage.value = '';
  isLoading.value = true;
  
  try {
    const results = await analyzeEnglishText(inputText.value);
    analysisResults.value = results;
  } catch (error) {
    console.error('分析文本时出错:', error);
    errorMessage.value = '分析文本时出错，请重试';
  } finally {
    isLoading.value = false;
  }
}
</script>

<template>
  <div class="container mx-auto px-4 py-10 max-w-4xl">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-6 mb-8">
      <h1 class="text-3xl font-bold mb-6 text-center text-gray-800">英文音标分析器</h1>
      
      <div class="mb-6">
        <label for="text-input" class="block text-sm font-medium text-gray-700 mb-2">输入英文文本：</label>
        <textarea 
          id="text-input"
          v-model="inputText"
          class="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          rows="4"
          placeholder="请输入英文文本，例如：Hello world. How are you?"
        ></textarea>
      </div>
      
      <div v-if="errorMessage" class="mb-4 text-red-500 text-sm">
        {{ errorMessage }}
      </div>
      
      <div class="flex justify-center mb-4">
        <button 
          @click="analyzeText"
          :disabled="isLoading"
          class="px-8 py-3 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all"
        >
          {{ isLoading ? '分析中...' : '分析文本' }}
        </button>
      </div>
    </div>
    
    <div v-if="analysisResults.length > 0" class="mt-8">
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <h2 class="text-xl font-semibold mb-6 text-gray-800">分析结果：</h2>
        
        <SentenceAnalysisComponent 
          v-for="(result, index) in analysisResults" 
          :key="index" 
          :words="result.words" 
          :sentence-index="index"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 可以添加自定义样式 */
</style>
