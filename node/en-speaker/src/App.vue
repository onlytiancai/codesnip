<script setup lang="ts">
import { ref } from 'vue';

// 导入模块
// @ts-ignore
import { Howl } from 'howler';
// 导入FFmpeg模块（使用动态导入方式）
let FFmpeg: any;
let ffmpegModuleLoaded = false;
let ffmpegModulePromise: Promise<void>;

// 实现fetchFile函数
const fetchFile = async (blob: Blob): Promise<Uint8Array> => {
  const arrayBuffer = await blob.arrayBuffer();
  return new Uint8Array(arrayBuffer);
};

// 动态导入FFmpeg模块
ffmpegModulePromise = import('@ffmpeg/ffmpeg').then(ffmpegModule => {
  // 使用类型断言来绕过TypeScript错误
  const module = ffmpegModule as any;
  FFmpeg = module.FFmpeg || module.default?.FFmpeg;
  ffmpegModuleLoaded = true;
  console.log('FFmpeg module loaded successfully:', { FFmpeg });
}).catch(error => {
  console.error('Failed to load FFmpeg module:', error);
  ffmpegModuleLoaded = false;
});

interface Sentence {
  id: number;
  text: string;
  audio: string | null;
  isRecording: boolean;
  isPlaying: boolean;
}

const inputText = ref("Postcards always spoil my holidays. Last summer, I went to Italy. I visited museums and sat in public gardens. ");
const sentences = ref<Sentence[]>([]);
const isProcessing = ref(false);
const isCreatingVideo = ref(false);
const videoProgress = ref(0);
const videoPreviewUrl = ref('');
let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];
let currentHowl: Howl | null = null;
let ffmpegInstance: any = null;

// wasm下载相关变量
const wasmDownloadProgress = ref(0);
const isDownloadingWasm = ref(false);
const wasmDownloadStatus = ref('');
const wasmTestStatus = ref('');
const wasmFileUrl = ref('');

// 下载wasm文件并显示进度
const downloadWasm = async () => {
  try {
    isDownloadingWasm.value = true;
    wasmDownloadStatus.value = '开始下载wasm文件...';
    wasmDownloadProgress.value = 0;
    
    const wasmUrl = 'https://webapp.ihuhao.com/cdn/@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.wasm';
    const response = await fetch(wasmUrl, {
      method: 'GET',
      headers: {
        'Accept': '*/*'
      }
    });
    
    if (!response.ok) {
      throw new Error(`下载失败: ${response.statusText}`);
    }
    
    const totalSize = parseInt(response.headers.get('content-length') || '0');
    let downloadedSize = 0;
    const reader = response.body?.getReader();
    
    if (!reader) {
      throw new Error('无法获取响应体');
    }
    
    const chunks: Uint8Array[] = [];
    
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        break;
      }
      
      chunks.push(value);
      downloadedSize += value.length;
      
      if (totalSize > 0) {
        wasmDownloadProgress.value = Math.round((downloadedSize / totalSize) * 100);
      }
    }
    
    const blob = new Blob(chunks as BlobPart[], { type: 'application/wasm' });
    const fileUrl = URL.createObjectURL(blob);
    wasmFileUrl.value = fileUrl;
    
    wasmDownloadStatus.value = 'wasm文件下载成功！';
    wasmDownloadProgress.value = 100;
    
    // 下载完成后自动测试wasm
    setTimeout(() => {
      testWasm();
    }, 500);
    
  } catch (error) {
    console.error('下载wasm文件失败:', error);
    wasmDownloadStatus.value = `下载失败: ${error instanceof Error ? error.message : '未知错误'}`;
  } finally {
    isDownloadingWasm.value = false;
  }
};

// 测试wasm是否能正常工作
const testWasm = async () => {
  try {
    wasmTestStatus.value = '开始测试wasm...';
    
    // 初始化FFmpeg实例
    const ffmpeg = await initFFmpeg();
    
    // 加载FFmpeg（使用本地下载的wasm文件）
    await ffmpeg.load({
      coreURL: 'https://webapp.ihuhao.com/cdn/@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.js',
      wasmURL: wasmFileUrl.value || 'https://webapp.ihuhao.com/cdn/@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.wasm'
    });
    
    // 简单测试：检查FFmpeg版本
    await ffmpeg.exec(['-version']);
    
    wasmTestStatus.value = 'wasm测试成功！FFmpeg加载正常。';
    
  } catch (error) {
    console.error('wasm测试失败:', error);
    wasmTestStatus.value = `测试失败: ${error instanceof Error ? error.message : '未知错误'}`;
  }
};

// 初始化FFmpeg实例
const initFFmpeg = async () => {
  // 等待FFmpeg模块加载完成
  if (!ffmpegModuleLoaded) {
    await ffmpegModulePromise;
  }
  
  if (!FFmpeg) {
    throw new Error('Failed to load FFmpeg module');
  }
  
  if (!ffmpegInstance) {
    ffmpegInstance = new FFmpeg({
      log: true,
      progress: (progress: any) => {
        console.log('FFmpeg progress callback:', progress);
      }
    });
  }
  
  return ffmpegInstance;
};

const processText = () => {
  if (!inputText.value.trim()) return;
  isProcessing.value = true;
  // 简单的分句逻辑，按句号、问号、感叹号分割
  const text = inputText.value.trim();
  const sentenceArray = text.split(/[.!?]+/).filter(s => s.trim());
  
  sentences.value = sentenceArray.map((sentence, index) => ({
    id: index + 1,
    text: sentence.trim(),
    audio: null,
    isRecording: false,
    isPlaying: false
  }));
  
  isProcessing.value = false;
};

const startRecording = async (sentenceId: number) => {
  try {
    // 停止之前可能正在进行的录音
    if (mediaRecorder) {
      mediaRecorder.stop();
    }
    
    // 停止之前可能正在播放的音频
    if (currentHowl) {
      currentHowl.stop();
      currentHowl = null;
    }
    
    // 获取用户媒体设备权限
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    
    // 更新句子状态
    const sentenceIndex = sentences.value.findIndex(s => s.id === sentenceId);
    if (sentenceIndex !== -1 && sentences.value[sentenceIndex]) {
      sentences.value[sentenceIndex].isRecording = true;
    }
    
    // 监听录音数据
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };
    
    // 录音结束时处理
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // 更新句子的音频数据
      const sentenceIndex = sentences.value.findIndex(s => s.id === sentenceId);
      if (sentenceIndex !== -1 && sentences.value[sentenceIndex]) {
        sentences.value[sentenceIndex].audio = audioUrl;
        sentences.value[sentenceIndex].isRecording = false;
      }
      
      // 停止媒体流
      stream.getTracks().forEach(track => track.stop());
    };
    
    // 开始录音
    mediaRecorder.start();
  } catch (error) {
    console.error('录音失败:', error);
    // 重置状态
    const sentenceIndex = sentences.value.findIndex(s => s.id === sentenceId);
    if (sentenceIndex !== -1 && sentences.value[sentenceIndex]) {
      sentences.value[sentenceIndex].isRecording = false;
    }
  }
};

const stopRecording = () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
  }
};

const playAudio = async (sentenceId: number) => {
  try {
    // 停止之前可能正在播放的音频
    if (currentHowl) {
      currentHowl.stop();
      currentHowl = null;
    }
    
    // 停止之前可能正在进行的录音
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }
    
    // 获取句子的音频
    const sentence = sentences.value.find(s => s.id === sentenceId);
    if (!sentence || !sentence.audio) return;
    
    // 更新播放状态
    sentence.isPlaying = true;
    
    // 使用Howler.js播放音频
    currentHowl = new Howl({
      src: [sentence.audio],
      format: ['wav'],
      onend: () => {
        sentence.isPlaying = false;
        currentHowl = null;
      },
      onplayerror: () => {
        console.error('播放失败');
        sentence.isPlaying = false;
        currentHowl = null;
      }
    });
    
    // 开始播放
    currentHowl.play();
  } catch (error) {
    console.error('播放失败:', error);
    // 重置状态
    const sentence = sentences.value.find(s => s.id === sentenceId);
    if (sentence) {
      sentence.isPlaying = false;
    }
  }
};

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
  startRecording(sentenceId);
};

// 生成静态图片（包含完整文本）
const generateImage = (): string => {
  const canvas = document.createElement('canvas');
  canvas.width = 720;
  canvas.height = 1280;
  const ctx = canvas.getContext('2d');
  
  if (!ctx) return '';
  
  // 绘制背景
  ctx.fillStyle = '#f0f4f8';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // 绘制边框
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 4;
  ctx.strokeRect(40, 40, canvas.width - 80, canvas.height - 80);
  
  // 绘制标题
  ctx.fillStyle = '#4f46e5';
  ctx.font = '48px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('英语朗读', canvas.width / 2, 100);
  
  // 绘制文本
  ctx.fillStyle = '#1e293b';
  ctx.font = '32px Arial';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  
  const text = inputText.value;
  const maxWidth = canvas.width - 120;
  const lineHeight = 40;
  const x = 60;
  let y = 160;
  
  // 文本换行
  const words = text.split(' ');
  let line = '';
  
  for (const word of words) {
    const testLine = line + word + ' ';
    const metrics = ctx.measureText(testLine);
    const testWidth = metrics.width;
    
    if (testWidth > maxWidth && line !== '') {
      ctx.fillText(line, x, y);
      line = word + ' ';
      y += lineHeight;
    } else {
      line = testLine;
    }
  }
  ctx.fillText(line, x, y);
  
  return canvas.toDataURL('image/png');
};

// 合并音频文件
const mergeAudioFiles = async (): Promise<Blob | null> => {
  try {
    // 检查是否所有句子都有录音
    const allRecorded = sentences.value.every(sentence => sentence.audio);
    if (!allRecorded) {
      alert('请为所有句子录制音频');
      return null;
    }
    
    // 初始化FFmpeg实例
    const ffmpeg = await initFFmpeg();
    
    // 加载FFmpeg
    if (!ffmpeg.loaded) {
      videoProgress.value = 10;
      await ffmpeg.load({
        coreURL: 'https://webapp.ihuhao.com/cdn/@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.js',
        wasmURL: 'https://webapp.ihuhao.com/cdn/@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.wasm'
      });
      videoProgress.value = 20;
    }
    
    // 为每个音频文件创建临时文件
    for (let i = 0; i < sentences.value.length; i++) {
      const sentence = sentences.value[i];
      if (sentence && sentence.audio) {
        const response = await fetch(sentence.audio);
        const blob = await response.blob();
        await ffmpeg.writeFile(`input${i}.wav`, await fetchFile(blob));
      }
    }
    
    videoProgress.value = 30;
    
    // 使用filter_complex方式合并音频，更可靠
    const inputArgs = [];
    const filterArgs = [];
    
    // 构建输入参数和滤镜参数
    for (let i = 0; i < sentences.value.length; i++) {
      inputArgs.push('-i', `input${i}.wav`);
      filterArgs.push(`[${i}:a]`);
    }
    
    // 构建完整的滤镜链
    filterArgs.push('concat=n=' + sentences.value.length + ':v=0:a=1[outa]');
    
    // 使用FFmpeg合并音频
    await ffmpeg.exec([
      ...inputArgs,
      '-filter_complex', filterArgs.join(''),
      '-map', '[outa]',
      'merged_audio.wav'
    ]);
    
    videoProgress.value = 50;
    
    // 读取合并后的音频
    const data = await ffmpeg.readFile('merged_audio.wav');
    const blob = new Blob([data.buffer], { type: 'audio/wav' });
    
    // 清理临时文件
    for (let i = 0; i < sentences.value.length; i++) {
      await ffmpeg.deleteFile(`input${i}.wav`);
    }
    
    return blob;
  } catch (error) {
    console.error('音频合并失败:', error);
    return null;
  }
};

// 合成视频
const createVideo = async () => {
  try {
    // 检查是否所有句子都有录音
    const allRecorded = sentences.value.every(sentence => sentence.audio);
    if (!allRecorded) {
      alert('请为所有句子录制音频');
      return;
    }
    
    isCreatingVideo.value = true;
    videoProgress.value = 0;
    videoPreviewUrl.value = '';
    
    // 生成图片
    const imageDataUrl = generateImage();
    if (!imageDataUrl) {
      alert('图片生成失败');
      isCreatingVideo.value = false;
      return;
    }
    
    videoProgress.value = 5;
    
    // 加载图片
    const imageResponse = await fetch(imageDataUrl);
    const imageBlob = await imageResponse.blob();
    
    // 合并音频
    const mergedAudioBlob = await mergeAudioFiles();
    if (!mergedAudioBlob) {
      isCreatingVideo.value = false;
      return;
    }
    
    videoProgress.value = 60;
    
    // 初始化FFmpeg实例
    const ffmpeg = await initFFmpeg();
    
    // 加载FFmpeg
    if (!ffmpeg.loaded) {
      await ffmpeg.load({
        coreURL: 'https://unpkg.com/@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.js',
        wasmURL: 'https://unpkg.com/@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.wasm'
      });
    }
    
    // 写入文件到FFmpeg虚拟文件系统
    await ffmpeg.writeFile('image.png', await fetchFile(imageBlob));
    await ffmpeg.writeFile('audio.wav', await fetchFile(mergedAudioBlob));
    
    videoProgress.value = 70;
    
    // 添加FFmpeg进度监听
    ffmpeg.on('progress', (progress: any) => {
      console.log('FFmpeg progress:', progress);
      if (progress && typeof progress === 'object') {
        // 计算实际进度百分比并保留两位小数
        const currentProgress = Math.round((70 + (progress.progress || 0) * 20) * 100) / 100;
        if (currentProgress > videoProgress.value && currentProgress < 90) {
          videoProgress.value = Math.min(currentProgress, 89.99);
        }
      }
    });
    
    // 使用FFmpeg生成视频
    await ffmpeg.exec([
      '-loop', '1',
      '-i', 'image.png',
      '-i', 'audio.wav',
      '-c:v', 'libx264',
      '-c:a', 'aac',
      '-shortest',
      '-pix_fmt', 'yuv420p',
      '-aspect', '9:16',
      '-progress', 'pipe:3',
      'output.mp4'
    ]);
    
    // 移除进度监听
    ffmpeg.off('progress');
    
    videoProgress.value = 90;
    
    // 读取生成的视频
    const data = await ffmpeg.readFile('output.mp4');
    const videoBlob = new Blob([data.buffer], { type: 'video/mp4' });
    const videoUrl = URL.createObjectURL(videoBlob);
    
    // 设置视频预览
    videoPreviewUrl.value = videoUrl;
    
    videoProgress.value = 100;
    
    // 清理临时文件
    await ffmpeg.deleteFile('image.png');
    await ffmpeg.deleteFile('audio.wav');
    await ffmpeg.deleteFile('merged_audio.wav');
    await ffmpeg.deleteFile('output.mp4');
    
    isCreatingVideo.value = false;
    
    // 显示成功消息
    alert('视频合成成功！您可以预览视频，然后下载到本地。');
    
  } catch (error) {
    console.error('视频合成失败:', error);
    alert('视频合成失败，请重试');
    isCreatingVideo.value = false;
    videoProgress.value = 0;
  }
};

// 下载视频到本地
const downloadVideo = () => {
  if (!videoPreviewUrl.value) return;
  
  const link = document.createElement('a');
  link.href = videoPreviewUrl.value;
  link.download = 'english-reading-video.mp4';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
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
          @click="processText" 
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
              @click="sentence.isRecording ? stopRecording() : startRecording(sentence.id)"
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
              @click="playAudio(sentence.id)"
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
          <!-- 下载wasm按钮和进度条 -->
          <div class="space-y-4">
            <h3 class="text-lg font-semibold text-gray-800">WASM文件管理</h3>
            <div class="flex justify-center">
              <button 
                @click="downloadWasm"
                :disabled="isDownloadingWasm"
                class="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {{ isDownloadingWasm ? '下载中...' : '下载WASM文件' }}
              </button>
            </div>
            
            <!-- WASM下载进度条 -->
            <div v-if="isDownloadingWasm || wasmDownloadProgress > 0" class="space-y-2">
              <div class="flex justify-between text-sm text-gray-600">
                <span>下载进度</span>
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
                @click="downloadVideo"
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
