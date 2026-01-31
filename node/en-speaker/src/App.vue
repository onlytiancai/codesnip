<script setup lang="ts">
import { ref } from 'vue';

// 导入类型
import type { Sentence } from './types';

// 导入服务函数
import { mergeAudioFiles, createVideoFromParts } from './services/ffmpeg';
import { startRecording, stopRecording, playAudio } from './services/audio';
import { processText, generateImage } from './services/text';
import { downloadWasm, testWasm } from './services/wasm';

// 导入工具函数
import { showToast, downloadVideo } from './utils/ui';

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
        <button 
          @click="handleProcessText" 
          :disabled="isProcessing"
          class="mt-4 px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {{ isProcessing ? '处理中...' : '开始朗读' }}
        </button>
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
          <div class="flex space-x-3">
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
              :disabled="!sentence.audio"
              :class="[
                'px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors',
                sentence.isPlaying 
                  ? 'bg-purple-600 text-white hover:bg-purple-700 focus:ring-purple-500' 
                  : 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500',
                !sentence.audio && 'opacity-50 cursor-not-allowed'
              ]"
            >
              {{ sentence.isPlaying ? '播放中...' : '回放' }}
            </button>
            <button 
              @click="reRecordAudio(sentence.id)"
              :disabled="sentence.isRecording || sentence.isPlaying"
              class="px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              重录
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
