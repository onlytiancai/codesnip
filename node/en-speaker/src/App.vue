<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';

// 导入类型
import type { Sentence, Voice, AudioCacheItem } from './types';

// 导入服务函数
import { mergeAudioFiles, createVideoFromParts } from './services/ffmpeg';
import { startRecording, stopRecording, playAudio } from './services/audio';
import { processText, generateImage } from './services/text';
import { downloadWasm, testWasm } from './services/wasm';

// 导入工具函数
import { showToast, downloadVideo } from './utils/ui';

// 动态导入TTS服务
let ttsService: any = null;
const loadTTSService = async () => {
  if (!ttsService) {
    const module = await import('./services/tts');
    ttsService = module.ttsService;
  }
  return ttsService;
};

// 响应式变量
const inputText = ref("Postcards always spoil my holidays. Last summer, I went to Italy. I visited museums and sat in public gardens. ");
const sentences = ref<Sentence[]>([]);
const isProcessing = ref(false);
const isCreatingVideo = ref(false);
const videoProgress = ref(0);
const videoPreviewUrl = ref('');

// wasm下载相关变量
const wasmDownloadProgress = ref(0);
const isDownloadingWasm = ref(false);
const wasmDownloadStatus = ref('');
const wasmTestStatus = ref('');
const wasmFileUrl = ref('');
const isWasmReady = ref(false);

// kokoro-js相关变量
const isLoadingModel = ref(false);
const modelLoadProgress = ref(0);
const modelLoadStatus = ref('');
const selectedVoice = ref('af_heart');
const availableVoices = ref<Voice[]>([]);
const isWebGPUSupported = ref(false);

// 语音缓存
const audioCache = ref<Record<string, Record<string, AudioCacheItem>>>({}); // 结构: { voiceId: { text: { audioUrl, timestamp } } }

// 清空缓存
const clearAudioCache = (voiceId?: string) => {
  if (voiceId) {
    // 只清空指定音色的缓存
    delete audioCache.value[voiceId];
    console.log(`[AI朗读] 清空音色 ${voiceId} 的语音缓存`);
  } else {
    // 清空所有缓存
    audioCache.value = {};
    console.log('[AI朗读] 清空所有语音缓存');
  }
};

// 处理文本
const handleProcessText = () => {
  if (!inputText.value.trim()) return;
  isProcessing.value = true;
  sentences.value = processText(inputText.value);
  isProcessing.value = false;
};

// 开始录音
const handleStartRecording = async (sentenceId: number) => {
  await startRecording(
    sentenceId,
    (id) => {
      const sentenceIndex = sentences.value.findIndex(s => s.id === id);
      if (sentenceIndex !== -1 && sentences.value[sentenceIndex]) {
        sentences.value[sentenceIndex].isRecording = true;
      }
    },
    (id, audioUrl) => {
      const sentenceIndex = sentences.value.findIndex(s => s.id === id);
      if (sentenceIndex !== -1 && sentences.value[sentenceIndex]) {
        sentences.value[sentenceIndex].audio = audioUrl;
        sentences.value[sentenceIndex].isRecording = false;
      }
    },
    (id) => {
      const sentenceIndex = sentences.value.findIndex(s => s.id === id);
      if (sentenceIndex !== -1 && sentences.value[sentenceIndex]) {
        sentences.value[sentenceIndex].isRecording = false;
      }
    }
  );
};

// 播放音频
const handlePlayAudio = async (sentenceId: number) => {
  const sentence = sentences.value.find(s => s.id === sentenceId);
  if (!sentence || !sentence.audio) return;
  
  await playAudio(
    sentenceId,
    sentence.audio,
    (id) => {
      const sentence = sentences.value.find(s => s.id === id);
      if (sentence) {
        sentence.isPlaying = true;
      }
    },
    (id) => {
      const sentence = sentences.value.find(s => s.id === id);
      if (sentence) {
        sentence.isPlaying = false;
      }
    },
    (id) => {
      const sentence = sentences.value.find(s => s.id === id);
      if (sentence) {
        sentence.isPlaying = false;
      }
    }
  );
};

// 重新录音
const reRecordAudio = (sentenceId: number) => {
  // 重置句子的音频数据
  const sentenceIndex = sentences.value.findIndex(s => s.id === sentenceId);
  if (sentenceIndex !== -1 && sentences.value[sentenceIndex]) {
    if (sentences.value[sentenceIndex].audio) {
      URL.revokeObjectURL(sentences.value[sentenceIndex].audio);
    }
    sentences.value[sentenceIndex].audio = null;
    sentences.value[sentenceIndex].isPlaying = false;
    sentences.value[sentenceIndex].isAiSpeaking = false;
  }
  
  // 开始重新录音
  handleStartRecording(sentenceId);
};

// 合成视频
const createVideo = async () => {
  try {
    // 检查是否所有句子都有录音
    const allRecorded = sentences.value.every(sentence => sentence.audio);
    if (!allRecorded) {
      showToast('请为所有句子录制音频', 'warning');
      return;
    }
    
    // 检查WASM是否就绪
    if (!isWasmReady.value) {
      showToast('正在准备WASM环境，请稍候...', 'info');
      videoProgress.value = 0;
      
      // 下载WASM文件
      isDownloadingWasm.value = true;
      try {
        wasmFileUrl.value = await downloadWasm(
          (progress) => {
            wasmDownloadProgress.value = progress;
          },
          (status) => {
            wasmDownloadStatus.value = status;
          }
        );
      } catch (error) {
        showToast('WASM文件下载失败，请重试', 'error');
        isDownloadingWasm.value = false;
        return;
      } finally {
        isDownloadingWasm.value = false;
      }
      
      // 测试WASM是否能正常工作
      const testResult = await testWasm(
        wasmFileUrl.value,
        (status) => {
          wasmTestStatus.value = status;
        }
      );
      
      if (!testResult) {
        showToast('WASM测试失败，请重试', 'error');
        return;
      }
      
      isWasmReady.value = true;
    }
    
    isCreatingVideo.value = true;
    videoProgress.value = 0;
    videoPreviewUrl.value = '';
    
    // 生成图片
    const imageDataUrl = generateImage(inputText.value);
    if (!imageDataUrl) {
      showToast('图片生成失败', 'error');
      isCreatingVideo.value = false;
      return;
    }
    
    videoProgress.value = 5;
    
    // 加载图片
    const imageResponse = await fetch(imageDataUrl);
    const imageBlob = await imageResponse.blob();
    
    // 合并音频
    const audioUrls = sentences.value.map(s => s.audio).filter((url): url is string => url !== null);
    const mergedAudioBlob = await mergeAudioFiles(audioUrls, wasmFileUrl.value);
    if (!mergedAudioBlob) {
      showToast('音频合并失败', 'error');
      isCreatingVideo.value = false;
      return;
    }
    
    videoProgress.value = 60;
    
    // 创建视频
    const videoBlob = await createVideoFromParts(imageBlob, mergedAudioBlob, wasmFileUrl.value);
    if (!videoBlob) {
      showToast('视频合成失败', 'error');
      isCreatingVideo.value = false;
      return;
    }
    
    videoProgress.value = 100;
    
    // 设置视频预览
    const videoUrl = URL.createObjectURL(videoBlob);
    videoPreviewUrl.value = videoUrl;
    
    isCreatingVideo.value = false;
    
    // 显示成功消息
    showToast('视频合成成功！您可以预览视频，然后下载到本地。', 'success');
    
  } catch (error) {
    console.error('视频合成失败:', error);
    showToast('视频合成失败，请重试', 'error');
    isCreatingVideo.value = false;
    videoProgress.value = 0;
  }
};

// 下载视频
const handleDownloadVideo = () => {
  downloadVideo(videoPreviewUrl.value);
};

// 检测WebGPU支持
const checkWebGPUSupport = async () => {
  const service = await loadTTSService();
  isWebGPUSupported.value = await service.checkWebGPUSupport();
};

// 加载Kokoro TTS模型
const loadKokoroModel = async () => {
  const service = await loadTTSService();
  if (service.isModelLoaded()) return;
  
  try {
    isLoadingModel.value = true;
    modelLoadProgress.value = 0;
    modelLoadStatus.value = '正在检测WebGPU支持...';
    
    // 检测WebGPU支持
    await checkWebGPUSupport();
    
    modelLoadStatus.value = `正在加载模型 (设备: ${isWebGPUSupported.value ? 'WebGPU' : 'WASM'})...`;
    
    // 加载模型并显示进度
    await service.loadModel((progress: number) => {
      modelLoadProgress.value = progress;
      modelLoadStatus.value = `加载中: ${progress}%`;
    });
    
    // 获取可用的语音
    availableVoices.value = service.getAvailableVoiceObjects();
    
    modelLoadStatus.value = '模型加载成功！';
    showToast('Kokoro TTS模型加载成功', 'success');
  } catch (error) {
    console.error('加载模型失败:', error);
    modelLoadStatus.value = '模型加载失败，请重试';
    showToast('加载模型失败，请重试', 'error');
  } finally {
    isLoadingModel.value = false;
  }
};

// 使用Kokoro TTS朗读句子
const speakSentence = async (sentenceId: number) => {
  const service = await loadTTSService();
  if (!service.isModelLoaded()) {
    showToast('请先加载TTS模型', 'warning');
    await loadKokoroModel();
    if (!service.isModelLoaded()) return;
  }
  
  const sentence = sentences.value.find(s => s.id === sentenceId);
  if (!sentence) return;
  
  try {
    sentence.isAiSpeaking = true;
    
    const voiceId = selectedVoice.value;
    const text = sentence.text;
    
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
      const audioBlob = await service.generateSpeech(text, voiceId);
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
    audioElement.onended = () => {
      sentence.isAiSpeaking = false;
      // 注意：由于缓存中需要使用audioUrl，这里不再revokeObjectURL
    };
    audioElement.onerror = () => {
      sentence.isAiSpeaking = false;
      // 注意：由于缓存中需要使用audioUrl，这里不再revokeObjectURL
    };
    
    await audioElement.play();
  } catch (error) {
    console.error('[AI朗读] 朗读失败:', error);
    showToast('朗读失败，请重试', 'error');
    sentence.isAiSpeaking = false;
  }
};

// 监听音色变化
watch(selectedVoice, (newVoice, oldVoice) => {
  if (oldVoice && newVoice !== oldVoice) {
    console.log(`[AI朗读] 音色从 ${oldVoice} 切换到 ${newVoice}`);
    // 切换音色时清空所有缓存，确保使用新音色生成语音
    clearAudioCache();
  }
});

// 组件挂载时检测WebGPU支持
onMounted(async () => {
  await checkWebGPUSupport();
});
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
    <div class="max-w-4xl mx-auto bg-white rounded-xl shadow-lg p-8">
      <h1 class="text-3xl font-bold text-indigo-600 mb-6 text-center">英语朗读App</h1>
      

      
      <div class="mb-8">
        <label for="text-input" class="block text-gray-700 font-medium mb-2">输入英文文本：</label>
        <textarea 
          id="text-input" 
          v-model="inputText" 
          class="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-colors"
          rows="6"
          placeholder="请输入英文文本，例如：Hello! How are you? I'm fine, thank you."
        ></textarea>
        <div class="mt-4 flex flex-col sm:flex-row gap-4">
          <button 
            @click="handleProcessText" 
            :disabled="isProcessing"
            class="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {{ isProcessing ? '处理中...' : '处理文本' }}
          </button>
          <button 
            @click="loadKokoroModel" 
            :disabled="isLoadingModel"
            class="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {{ isLoadingModel ? '加载中...' : '加载语音模型' }}
          </button>
        </div>
        
        <!-- 语音选择 -->
        <div v-if="availableVoices.length > 0" class="mt-4">
          <label for="voice-select" class="block text-gray-700 font-medium mb-2">选择语音：</label>
          <select 
            id="voice-select" 
            v-model="selectedVoice" 
            class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-colors"
          >
            <option v-for="voice in availableVoices" :key="voice.id" :value="voice.id">
              {{ voice.displayName }}
            </option>
          </select>
        </div>
        
        <!-- 模型加载进度 -->
        <div v-if="isLoadingModel || modelLoadProgress > 0" class="mt-4 space-y-2">
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
      
      <div v-if="sentences.length > 0" class="space-y-6">
        <h2 class="text-xl font-semibold text-gray-800 mb-4">句子列表</h2>
        <div 
          v-for="sentence in sentences" 
          :key="sentence.id"
          class="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
        >
          <div class="flex items-start justify-between mb-4">
            <div class="flex-1">
              <span class="inline-flex items-center justify-center w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 font-medium mr-3">
                {{ sentence.id }}
              </span>
              <span class="text-gray-800">{{ sentence.text }}</span>
            </div>
          </div>
          <div class="flex space-x-3 flex-wrap">
            <button 
              @click="sentence.isRecording ? stopRecording() : handleStartRecording(sentence.id)"
              :class="[
                'px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors',
                sentence.isRecording 
                  ? 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500' 
                  : 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500'
              ]"
            >
              {{ sentence.isRecording ? '停止录音' : '录音' }}
            </button>
            <button 
              @click="handlePlayAudio(sentence.id)"
              :disabled="!sentence.audio || sentence.isPlaying"
              :class="[
                'px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors',
                sentence.isPlaying 
                  ? 'bg-purple-600 text-white hover:bg-purple-700 focus:ring-purple-500' 
                  : 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500',
                (!sentence.audio || sentence.isPlaying) && 'opacity-50 cursor-not-allowed'
              ]"
            >
              {{ sentence.isPlaying ? '播放中...' : '回放' }}
            </button>
            <button 
              @click="reRecordAudio(sentence.id)"
              :disabled="sentence.isRecording || sentence.isPlaying || sentence.isAiSpeaking"
              class="px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              重录
            </button>
            <button 
              @click="speakSentence(sentence.id)"
              :disabled="sentence.isAiSpeaking"
              :class="[
                'px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors',
                sentence.isAiSpeaking 
                  ? 'bg-indigo-600 text-white hover:bg-indigo-700 focus:ring-indigo-500' 
                  : 'bg-purple-600 text-white hover:bg-purple-700 focus:ring-purple-500',
                sentence.isAiSpeaking && 'opacity-50 cursor-not-allowed'
              ]"
            >
              {{ sentence.isAiSpeaking ? 'AI朗读中...' : 'AI朗读' }}
            </button>
          </div>
        </div>
        
        <div class="mt-8 space-y-6">
          <!-- WASM状态显示 -->
          <div class="space-y-4" v-if="isDownloadingWasm || wasmDownloadStatus || wasmTestStatus">
            <h3 class="text-lg font-semibold text-gray-800">WASM状态</h3>
            
            <!-- WASM下载进度条 -->
            <div v-if="isDownloadingWasm || wasmDownloadProgress > 0" class="space-y-2">
              <div class="flex justify-between text-sm text-gray-600">
                <span>WASM下载进度</span>
                <span>{{ wasmDownloadProgress }}%</span>
              </div>
              <div class="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  class="bg-blue-600 h-2.5 rounded-full transition-all duration-300 ease-in-out"
                  :style="{ width: wasmDownloadProgress + '%' }"
                ></div>
              </div>
            </div>
            
            <!-- WASM下载状态 -->
            <div v-if="wasmDownloadStatus" class="text-sm" :class="wasmDownloadStatus.includes('成功') ? 'text-green-600' : wasmDownloadStatus.includes('失败') ? 'text-red-600' : 'text-blue-600'">
              {{ wasmDownloadStatus }}
            </div>
            
            <!-- WASM测试状态 -->
            <div v-if="wasmTestStatus" class="text-sm" :class="wasmTestStatus.includes('成功') ? 'text-green-600' : wasmTestStatus.includes('失败') ? 'text-red-600' : 'text-blue-600'">
              {{ wasmTestStatus }}
            </div>
          </div>
          
          <!-- 合成视频按钮 -->
          <div class="flex justify-center">
            <button 
              @click="createVideo"
              :disabled="sentences.length === 0 || sentences.some(s => !s.audio) || isCreatingVideo"
              class="px-8 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {{ isCreatingVideo ? '合成中...' : '合成视频' }}
            </button>
          </div>
          
          <!-- 合成进度条 -->
          <div v-if="isCreatingVideo" class="space-y-2">
            <div class="flex justify-between text-sm text-gray-600">
              <span>合成进度</span>
              <span>{{ videoProgress }}%</span>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-2.5">
              <div 
                class="bg-purple-600 h-2.5 rounded-full transition-all duration-300 ease-in-out"
                :style="{ width: videoProgress + '%' }"
              ></div>
            </div>
          </div>
          
          <!-- 视频预览和下载 -->
          <div v-if="videoPreviewUrl" class="space-y-4">
            <h3 class="text-lg font-semibold text-gray-800">视频预览</h3>
            <div class="flex justify-center bg-gray-900 rounded-lg overflow-hidden">
              <video 
                :src="videoPreviewUrl" 
                controls 
                class="w-auto max-w-full h-[500px] aspect-[9/16]"
                autoplay 
                muted 
                loop
              ></video>
            </div>
            <div class="flex justify-center">
              <button 
              @click="handleDownloadVideo"
              class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
            >
              下载视频到本地
            </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
</style>
