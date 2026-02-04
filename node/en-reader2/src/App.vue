<script setup lang="ts">
import { ref, watch, onMounted } from 'vue';
import { analyzeEnglishText } from './utils/phonemizer';
import type { SentenceAnalysis } from './utils/phonemizer';
import SentenceAnalysisComponent from './components/SentenceAnalysis.vue';
import { ttsService } from './services/tts';
import { translateService } from './services/translate';

const inputText = ref(`Last week I went to the theatre. I had a very good seat. The play was very interesting. I did not enjoy it. A young man and a young woman were sitting behind me. They were talking loudly. I got very angry. I could not hear the actors. I turned round. I looked at the man and the woman angrily. They did not pay any attention. In the end, I could not bear it. I turned round again. 'I can't hear a word!' I said angrily.

'It's none of your business,' the young man said rudely. 'This is a private conversation!'`);
// 扩展SentenceAnalysis类型，添加isAiSpeaking和translation属性
interface ExtendedSentenceAnalysis extends SentenceAnalysis {
  isAiSpeaking?: boolean;
  translation?: string;
  isTranslating?: boolean;
}

const analysisResults = ref<ExtendedSentenceAnalysis[]>([]);
const isLoading = ref(false);
const errorMessage = ref('');
const showPhonemes = ref(true);
const alignPhonemesWithWords = ref(true);
const autoTranslate = ref(false);

// TTS相关变量
const isLoadingModel = ref(false);
const modelLoadProgress = ref(0);
const modelLoadStatus = ref('');
const selectedVoice = ref('af_heart');
const availableVoices = ref<any[]>([]);
const isWebGPUSupported = ref(false);

// 当前播放状态
const currentAudioElement = ref<HTMLAudioElement | null>(null);
const currentSpeakingSentenceIndex = ref<number | null>(null);

// 语音缓存
const audioCache = ref<Record<string, Record<string, { audioUrl: string; timestamp: number }>>>({}); // 结构: { voiceId: { text: { audioUrl, timestamp } } }

async function analyzeText() {
  if (!inputText.value.trim()) {
    errorMessage.value = '请输入英文文本';
    return;
  }
  
  errorMessage.value = '';
  isLoading.value = true;
  
  try {
    const results = await analyzeEnglishText(inputText.value);
    // 为每个句子添加isAiSpeaking属性
    analysisResults.value = results.map(result => ({
      ...result,
      isAiSpeaking: false,
      translation: undefined,
      isTranslating: false
    }));
    
    // 如果开启了自动翻译，翻译所有句子
    if (autoTranslate.value) {
      await translateAllSentences();
    }
  } catch (error) {
    console.error('分析文本时出错:', error);
    errorMessage.value = '分析文本时出错，请重试';
  } finally {
    isLoading.value = false;
  }
}

// 翻译单个句子
async function translateSentence(sentenceIndex: number) {
  const sentence = analysisResults.value[sentenceIndex];
  if (!sentence || sentence.isTranslating) return;
  
  try {
    sentence.isTranslating = true;
    const text = sentence.words.map(word => word.word).join(' ');
    const translation = await translateService.translate(text);
    sentence.translation = translation;
  } catch (error) {
    console.error('翻译句子时出错:', error);
    showToast('谷歌翻译服务不可用，请检查网络连接', 'error');
  } finally {
    sentence.isTranslating = false;
  }
}

// 翻译所有句子
async function translateAllSentences() {
  for (let i = 0; i < analysisResults.value.length; i++) {
    await translateSentence(i);
  }
}

// 清空音频缓存
function clearAudioCache(voiceId?: string) {
  if (voiceId) {
    // 只清空指定音色的缓存
    delete audioCache.value[voiceId];
    console.log(`[AI朗读] 清空音色 ${voiceId} 的语音缓存`);
  } else {
    // 清空所有缓存
    audioCache.value = {};
    console.log('[AI朗读] 清空所有语音缓存');
  }
}

// 检测WebGPU支持
async function checkWebGPUSupport() {
  isWebGPUSupported.value = await ttsService.checkWebGPUSupport();
}

// 加载Kokoro TTS模型
async function loadKokoroModel() {
  if (ttsService.isModelLoaded()) return;
  
  try {
    isLoadingModel.value = true;
    modelLoadProgress.value = 0;
    modelLoadStatus.value = '正在检测WebGPU支持...';
    
    // 检测WebGPU支持
    await checkWebGPUSupport();
    
    modelLoadStatus.value = `正在加载模型 (设备: ${isWebGPUSupported.value ? 'WebGPU' : 'WASM'})...`;
    
    // 加载模型并显示进度
    await ttsService.loadModel((progress: number) => {
      modelLoadProgress.value = progress;
      modelLoadStatus.value = `加载中: ${progress}%`;
    });
    
    // 获取可用的语音
    availableVoices.value = ttsService.getAvailableVoiceObjects();
    
    modelLoadStatus.value = '模型加载成功！';
    showToast('Kokoro TTS模型加载成功', 'success');
  } catch (error) {
    console.error('加载模型失败:', error);
    modelLoadStatus.value = '模型加载失败，请重试';
    showToast('加载模型失败，请重试', 'error');
  } finally {
    isLoadingModel.value = false;
  }
}

// 显示提示信息
function showToast(message: string, type: 'success' | 'error' | 'warning' | 'info' = 'info') {
  // 简单的提示实现，实际项目中可以使用更复杂的组件
  const toast = document.createElement('div');
  toast.className = `fixed top-4 right-4 px-4 py-2 rounded-lg shadow-lg z-50 ${type === 'success' ? 'bg-green-500 text-white' : type === 'error' ? 'bg-red-500 text-white' : type === 'warning' ? 'bg-yellow-500 text-white' : 'bg-blue-500 text-white'}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  
  setTimeout(() => {
    toast.classList.add('opacity-0', 'transition-opacity', 'duration-300');
    setTimeout(() => {
      document.body.removeChild(toast);
    }, 300);
  }, 3000);
}

// 监听音色变化
watch(selectedVoice, (newVoice, oldVoice) => {
  if (oldVoice && newVoice !== oldVoice) {
    console.log(`[AI朗读] 音色从 ${oldVoice} 切换到 ${newVoice}`);
    // 切换音色时清空所有缓存，确保使用新音色生成语音
    clearAudioCache();
  }
});

// 处理句子朗读
async function handleSpeak(sentenceIndex: number) {
  const sentence = analysisResults.value[sentenceIndex];
  if (!sentence) return;
  
  // 检查模型是否已加载
  if (!ttsService.isModelLoaded()) {
    showToast('请先加载TTS模型', 'warning');
    await loadKokoroModel();
    if (!ttsService.isModelLoaded()) return;
  }
  
  try {
    // 停止当前正在播放的音频
    if (currentAudioElement.value) {
      currentAudioElement.value.pause();
      currentAudioElement.value.currentTime = 0;
      // 重置之前句子的状态
      if (currentSpeakingSentenceIndex.value !== null) {
        const previousSentence = analysisResults.value[currentSpeakingSentenceIndex.value];
        if (previousSentence) {
          previousSentence.isAiSpeaking = false;
        }
      }
    }
    
    // 更新当前播放状态
    sentence.isAiSpeaking = true;
    currentSpeakingSentenceIndex.value = sentenceIndex;
    
    const voiceId = selectedVoice.value;
    // 构建完整句子文本
    const text = sentence.words.map(word => word.word).join(' ');
    
    // 检查缓存
    if (!audioCache.value[voiceId]) {
      audioCache.value[voiceId] = {};
    }
    
    let audioUrl: string;
    if (audioCache.value[voiceId][text]) {
      // 使用缓存的音频
      audioUrl = audioCache.value[voiceId][text].audioUrl;
      console.log(`[AI朗读] 使用缓存的语音: ${text.substring(0, 20)}...`);
    } else {
      // 生成新音频
      console.log(`[AI朗读] 生成新语音: ${text.substring(0, 20)}...`);
      const audioBlob = await ttsService.generateSpeech(text, voiceId);
      audioUrl = URL.createObjectURL(audioBlob);
      
      // 存储到缓存
      audioCache.value[voiceId][text] = {
        audioUrl,
        timestamp: Date.now()
      };
      console.log(`[AI朗读] 语音已缓存: ${text.substring(0, 20)}...`);
    }
    
    // 播放音频
    const audioElement = new Audio(audioUrl);
    currentAudioElement.value = audioElement;
    
    audioElement.onended = () => {
      sentence.isAiSpeaking = false;
      currentAudioElement.value = null;
      currentSpeakingSentenceIndex.value = null;
      // 注意：由于缓存中需要使用audioUrl，这里不再revokeObjectURL
    };
    audioElement.onerror = () => {
      sentence.isAiSpeaking = false;
      currentAudioElement.value = null;
      currentSpeakingSentenceIndex.value = null;
      // 注意：由于缓存中需要使用audioUrl，这里不再revokeObjectURL
    };
    
    await audioElement.play();
  } catch (error) {
    console.error('[AI朗读] 朗读失败:', error);
    showToast('朗读失败，请重试', 'error');
    sentence.isAiSpeaking = false;
    currentAudioElement.value = null;
    currentSpeakingSentenceIndex.value = null;
  }
}

// 组件挂载时检测WebGPU支持
onMounted(async () => {
  await checkWebGPUSupport();
});
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
      
      <!-- 音标显示选项 -->
      <div class="mb-6 space-y-3">
        <div class="flex items-center">
          <input 
            id="show-phonemes"
            v-model="showPhonemes"
            type="checkbox"
            class="w-4 h-4 text-blue-600 rounded focus:ring-blue-500 border-gray-300"
          >
          <label for="show-phonemes" class="ml-2 block text-sm text-gray-700">
            显示音标
          </label>
        </div>
        
        <div class="flex items-center" v-if="showPhonemes">
          <input 
            id="align-phonemes"
            v-model="alignPhonemesWithWords"
            type="checkbox"
            class="w-4 h-4 text-blue-600 rounded focus:ring-blue-500 border-gray-300"
          >
          <label for="align-phonemes" class="ml-2 block text-sm text-gray-700">
            音标与单词对齐
          </label>
        </div>
        
        <div class="flex items-center">
          <input 
            id="auto-translate"
            v-model="autoTranslate"
            type="checkbox"
            class="w-4 h-4 text-blue-600 rounded focus:ring-blue-500 border-gray-300"
          >
          <label for="auto-translate" class="ml-2 block text-sm text-gray-700">
            自动翻译
          </label>
        </div>
      </div>
      
      <div v-if="errorMessage" class="mb-4 text-red-500 text-sm">
        {{ errorMessage }}
      </div>
      
      <div class="flex flex-col sm:flex-row gap-4 justify-center mb-4">
        <button 
          @click="analyzeText"
          :disabled="isLoading"
          class="px-8 py-3 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all"
        >
          {{ isLoading ? '分析中...' : '分析文本' }}
        </button>
        <button 
          @click="loadKokoroModel" 
          :disabled="isLoadingModel"
          class="px-6 py-3 bg-green-600 text-white font-medium rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all"
        >
          {{ isLoadingModel ? '加载中...' : '加载语音模型' }}
        </button>
      </div>
      
      <!-- 语音选择 -->
      <div v-if="availableVoices.length > 0" class="mb-6">
        <label for="voice-select" class="block text-sm font-medium text-gray-700 mb-2">选择语音：</label>
        <select 
          id="voice-select" 
          v-model="selectedVoice" 
          class="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
        >
          <option v-for="voice in availableVoices" :key="voice.id" :value="voice.id">
            {{ voice.displayName }}
          </option>
        </select>
      </div>
      
      <!-- 模型加载进度 -->
      <div v-if="isLoadingModel || modelLoadProgress > 0" class="mb-6 space-y-2">
        <div class="flex justify-between text-sm text-gray-600">
          <span>模型加载进度</span>
          <span>{{ modelLoadProgress }}%</span>
        </div>
        <div class="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            class="bg-green-600 h-2.5 rounded-full transition-all duration-300 ease-in-out"
            :style="{ width: modelLoadProgress + '%' }"
          ></div>
        </div>
        <div v-if="modelLoadStatus" class="text-sm" :class="modelLoadStatus.includes('成功') ? 'text-green-600' : modelLoadStatus.includes('失败') ? 'text-red-600' : 'text-blue-600'">
          {{ modelLoadStatus }}
        </div>
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
          :show-phonemes="showPhonemes"
          :align-phonemes-with-words="alignPhonemesWithWords"
          :is-ai-speaking="result.isAiSpeaking"
          :translation="result.translation"
          :is-translating="result.isTranslating"
          @speak="handleSpeak"
          @translate="translateSentence"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 可以添加自定义样式 */
</style>
