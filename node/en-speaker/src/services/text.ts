import type { Sentence } from '../types';

// 处理文本，将其分割为句子
export const processText = (text: string): Sentence[] => {
  if (!text.trim()) return [];
  
  // 简单的分句逻辑，按句号、问号、感叹号分割
  const trimmedText = text.trim();
  const sentenceArray = trimmedText.split(/[.!?]+/).filter(s => s.trim());
  
  return sentenceArray.map((sentence, index) => ({
    id: index + 1,
    text: sentence.trim(),
    audio: null,
    isRecording: false,
    isPlaying: false,
    isAiSpeaking: false
  }));
};

// 生成静态图片（包含完整文本）
export const generateImage = (text: string): string => {
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
