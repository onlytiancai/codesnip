import * as ort from "onnxruntime-web";
import { KokoroTTS } from "kokoro-js";
import { env } from "@huggingface/transformers";

// 1. 配 WASM CDN
const WASM_BASE = "https://cdn.jsdmirror.com/npm/@huggingface/transformers@3.8.1/dist/";

// 2. 配模型下载地址
const MODEL_HOST = "https://webapp.ihuhao.com/cdn/";
const MODEL_PATH_TEMPLATE = "{model}/";

// 同时设置根版本和 transformers 版本的 wasmPaths
ort.env.wasm.wasmPaths = WASM_BASE;
if (env.backends.onnx?.wasm) {
  env.backends.onnx.wasm.wasmPaths = WASM_BASE;
}

// 自定义模型下载地址
env.remoteHost = MODEL_HOST;
env.remotePathTemplate = MODEL_PATH_TEMPLATE;




// 检测WebGPU支持
export async function detectWebGPU(): Promise<boolean> {
  try {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      return false;
    }
    const adapter = await (navigator as any).gpu.requestAdapter();
    return !!adapter;
  } catch (e) {
    return false;
  }
}

// TTS服务类
export class TTSService {
  private tts: KokoroTTS | null = null;
  private availableVoices: Record<string, any> = {};
  private isWebGPUSupported: boolean = false;

  // 检测WebGPU支持
  async checkWebGPUSupport(): Promise<boolean> {
    this.isWebGPUSupported = await detectWebGPU();
    return this.isWebGPUSupported;
  }

  // 加载Kokoro TTS模型
  async loadModel(progressCallback?: (progress: number) => void): Promise<KokoroTTS> {
    if (this.tts) return this.tts;

    try {
      // 检测WebGPU支持
      this.isWebGPUSupported = await detectWebGPU();
      
      const device = this.isWebGPUSupported ? 'webgpu' : 'wasm';
      const dtype = this.isWebGPUSupported ? 'fp32' : 'q8';

      // 加载模型并显示进度
      const model_id = 'onnx-community/Kokoro-82M-v1.0-ONNX';
      this.tts = await KokoroTTS.from_pretrained(model_id, {
        dtype,
        device,
        progress_callback: (progress) => {
          // 确保进度值是有效的数字
          const validProgress = typeof progress === 'number' && !isNaN(progress) ? progress : 0;
          const percentage = Math.max(0, Math.min(100, Math.round(validProgress * 100)));
          if (progressCallback) {
            progressCallback(percentage);
          }
        }
      });

      // 获取可用的语音
      try {
        if (this.tts && 'voices' in this.tts) {
          this.availableVoices = this.tts.voices;
          console.log('可用语音列表:', this.availableVoices);
        }
      } catch (error) {
        console.error('获取语音列表失败:', error);
        this.availableVoices = {};
      }

      return this.tts;
    } catch (error) {
      console.error('加载模型失败:', error);
      throw error;
    }
  }

  // 获取TTS实例
  getTTS(): KokoroTTS | null {
    return this.tts;
  }

  // 获取原始语音对象映射
  getRawVoices(): Record<string, any> {
    return this.availableVoices;
  }

  // 获取可用语音对象数组
  getAvailableVoiceObjects(): Array<{
    id: string;
    name: string;
    gender: string;
    language: string;
    overallGrade: string;
    targetQuality: string;
    displayName: string;
  }> {
    return Object.entries(this.availableVoices).map(([id, voice]) => {
      const voiceObj = voice as any;
      // 构建显示名称，格式为 "Name (Language Gender)"
      let languageDisplay = "";
      if (voiceObj.language === "en-us") {
        languageDisplay = "American";
      } else if (voiceObj.language === "en-gb") {
        languageDisplay = "British";
      } else {
        languageDisplay = voiceObj.language;
      }
      
      const displayName = `${voiceObj.name || id} (${languageDisplay} ${voiceObj.gender || ""})`;
      
      return {
        id,
        name: voiceObj.name || id,
        gender: voiceObj.gender || "",
        language: voiceObj.language || "",
        overallGrade: voiceObj.overallGrade || "",
        targetQuality: voiceObj.targetQuality || "",
        displayName
      };
    });
  }

  // 为保持兼容性，保留原方法
  getAvailableVoiceNames(): string[] {
    return Object.keys(this.availableVoices);
  }

  // 检查模型是否已加载
  isModelLoaded(): boolean {
    return !!this.tts;
  }

  // 生成语音
  async generateSpeech(text: string, voice: string): Promise<Blob> {
    if (!this.tts) {
      throw new Error('TTS模型未加载');
    }

    try {
      const audio = await this.tts.generate(text, {
        voice: voice as any // 类型断言，避免类型错误
      });

      // 使用audio.toBlob()方法获取Blob，参考demo的实现
      if (audio && typeof audio === 'object' && 'toBlob' in audio && typeof audio.toBlob === 'function') {
        return await audio.toBlob();
      }
      
      // 如果无法处理音频对象，抛出错误
      throw new Error('无法处理音频对象，未知的音频格式');
    } catch (error) {
      console.error('生成语音失败:', error);
      throw error;
    }
  }
}

// 导出单例实例
export const ttsService = new TTSService();
