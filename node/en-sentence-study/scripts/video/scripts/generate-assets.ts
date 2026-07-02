// 资产生成编排脚本
// 用法：pnpm exec tsx scripts/generate-assets.ts
//
// 流程：
//   1. 调用 TTS 模块生成中英混合音频 → public/audio/hello.mp3
//   2. 调用 image 模块生成场景插画 → public/images/scene.png

import { writeFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { synthesize } from '../src/api/minimax-tts';
import { generate, writeBase64ToFile } from '../src/api/minimax-image';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

// ─── 配置 ───────────────────────────────────────────────
const TTS_TEXT = 'Hello, 欢迎使用 Remotion 英语口语视频生成器，环境测试运行成功。';
const IMAGE_PROMPT =
  'A bright, minimalist flat illustration of an open book and a small rocket launching, ' +
  'symbolizing learning and creation. Soft pastel colors (mint green, light yellow, peach). ' +
  'White background, no text, no real people, no watermark, 9:16 portrait composition, ' +
  'vector art style, cheerful mood.';

// ─── TTS ────────────────────────────────────────────────
async function genTts() {
  console.log('🎤 生成 TTS...');
  console.log(`   text: ${TTS_TEXT}`);
  const { buffer, durationMs } = await synthesize({
    text: TTS_TEXT,
    language: 'zh',
  });
  const outPath = resolve(ROOT, 'public/audio/hello.mp3');
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, buffer);
  console.log(`   ✅ ${outPath}（${(buffer.length / 1024).toFixed(1)} KB，${(durationMs / 1000).toFixed(2)}s）`);
}

// ─── Image ──────────────────────────────────────────────
async function genImage() {
  console.log('🎨 生成 AI 插画...');
  console.log(`   prompt: ${IMAGE_PROMPT}`);
  const imgs = await generate({
    prompt: IMAGE_PROMPT,
    aspectRatio: '9:16',
    model: 'image-01',
    responseFormat: 'base64', // base64 直存，避开 URL 签名过期
  });
  const img = imgs[0];
  console.log(`   base64 length: ${img.base64?.length ?? 0} chars`);

  const outPath = resolve(ROOT, 'public/images/scene.jpg');
  writeBase64ToFile(img, outPath);
  console.log(`   ✅ ${outPath}`);
}

async function main() {
  if (!process.env.MINIMAX_API_KEY) {
    console.error('❌ MINIMAX_API_KEY 未设置');
    process.exit(1);
  }
  console.log('='.repeat(60));
  console.log('🎬 Hello Remotion · 资产生成');
  console.log('='.repeat(60));
  await genTts();
  await genImage();
  console.log('='.repeat(60));
  console.log('✅ 全部完成');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});