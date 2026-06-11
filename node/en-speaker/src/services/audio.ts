// @ts-ignore
import { Howl } from 'howler';

let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];
let currentHowl: Howl | null = null;

// 开始录音
export const startRecording = async (sentenceId: number, onRecordingStart: (id: number) => void, onRecordingStop: (id: number, audioUrl: string) => void, onError: (id: number) => void) => {
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
    
    // 通知录音开始
    onRecordingStart(sentenceId);
    
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
      
      // 通知录音停止并返回音频URL
      onRecordingStop(sentenceId, audioUrl);
      
      // 停止媒体流
      stream.getTracks().forEach(track => track.stop());
    };
    
    // 开始录音
    mediaRecorder.start();
  } catch (error) {
    console.error('录音失败:', error);
    // 通知错误
    onError(sentenceId);
  }
};

// 停止录音
export const stopRecording = () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
  }
};

// 播放音频
export const playAudio = async (sentenceId: number, audioUrl: string, onPlayStart: (id: number) => void, onPlayEnd: (id: number) => void, onError: (id: number) => void) => {
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
    
    if (!audioUrl) return;
    
    // 通知播放开始
    onPlayStart(sentenceId);
    
    // 使用Howler.js播放音频
    currentHowl = new Howl({
      src: [audioUrl],
      format: ['wav'],
      onend: () => {
        onPlayEnd(sentenceId);
        currentHowl = null;
      },
      onplayerror: () => {
        console.error('播放失败');
        onError(sentenceId);
        currentHowl = null;
      }
    });
    
    // 开始播放
    currentHowl.play();
  } catch (error) {
    console.error('播放失败:', error);
    // 通知错误
    onError(sentenceId);
  }
};

// 释放音频资源
export const releaseAudioResources = () => {
  if (currentHowl) {
    currentHowl.stop();
    currentHowl = null;
  }
  if (mediaRecorder) {
    mediaRecorder.stop();
    mediaRecorder = null;
  }
  audioChunks = [];
};
