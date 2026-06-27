// 全量资产生成：根据 desc JSON 生成所有卡片的 TTS 音频 + 场景 AI 插画
// 用法：
//   pnpm exec tsx scripts/generate-all.ts                              # 默认 desc/1.draft.json
//   pnpm exec tsx scripts/generate-all.ts path/to/desc.json            # 指定 desc
//   pnpm exec tsx scripts/generate-all.ts path/to/desc.json --force    # 强制重新生成所有资产
//
// 复用：
//   - synthesize() from ../src/api/minimax-tts        (Step 6 验证)
//   - generate() + writeBase64ToFile() from ../src/api/minimax-image (Step 3 验证)
//   - duration_sec 公式与 generate-card-audio.ts 一致：max(4, ceil(sum_ms/1000) + 1)
//
// 输出：
//   - 音频：public/audio/<desc-id>/<cardIdx>-<segIdx>-<lang>.mp3
//   - 图片：public/images/<desc-id>.jpg
//   - 回写 desc JSON：填充 tts_segments[].audio_path/duration_ms、cards[].duration_sec
//                    scene_image.{prompt,url,local_path}，重算顶层 duration_sec/frames

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execSync } from 'node:child_process';

import { synthesize } from '../src/api/minimax-tts';
import { generate, writeBase64ToFile } from '../src/api/minimax-image';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

const DEFAULT_DESC = resolve(ROOT, 'scripts/desc/1.draft.json');
const AUDIO_BASE = resolve(ROOT, 'public/audio');
const IMAGE_BASE = resolve(ROOT, 'public/images');

const FORCE = process.argv.includes('--force');
const MIN_DURATION_SEC = 4;
const PADDING_SEC = 1;

// ─── 复用：duration 公式 + 跳过已存在文件 ─────────────────────────
function measureMs(audioPath: string, fallbackTextLength: number): number {
  try {
    const out = execSync(
      `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${audioPath}"`,
      { encoding: 'utf-8' }
    );
    return Math.round(parseFloat(out.trim()) * 1000);
  } catch {
    // 无 ffprobe 时按 4 字/秒保守估算
    return fallbackTextLength * 250;
  }
}

async function generateCardAudio(
  descId: string,
  card: any
): Promise<{ totalMs: number; durationSec: number; regenerated: number; skipped: number }> {
  const cardAudioDir = join(AUDIO_BASE, descId);
  mkdirSync(cardAudioDir, { recursive: true });

  let totalMs = 0;
  let regenerated = 0;
  let skipped = 0;

  for (let i = 0; i < card.tts_segments.length; i++) {
    const seg = card.tts_segments[i];
    const outName = `${card.index}-${i}-${seg.lang}.mp3`;
    const outPath = join(cardAudioDir, outName);

    if (!FORCE && existsSync(outPath)) {
      const ms = measureMs(outPath, seg.text.length);
      seg.audio_path = outPath;
      seg.duration_ms = ms;
      totalMs += ms;
      skipped++;
      console.log(`   ↻ card ${card.index} seg ${i} (${seg.lang}) 已存在 · ${(ms / 1000).toFixed(2)}s`);
      continue;
    }

    console.log(`   🎤 card ${card.index} seg ${i} (${seg.lang}): "${seg.text}"`);
    const { buffer, durationMs } = await synthesize({
      text: seg.text,
      language: seg.lang,
    });
    writeFileSync(outPath, buffer);
    seg.audio_path = outPath;
    seg.duration_ms = durationMs;
    totalMs += durationMs;
    regenerated++;
    console.log(`      ✓ ${(buffer.length / 1024).toFixed(1)} KB · ${(durationMs / 1000).toFixed(2)}s`);
  }

  const durationSec = Math.max(MIN_DURATION_SEC, Math.ceil(totalMs / 1000) + PADDING_SEC);
  card.duration_sec = durationSec;
  return { totalMs, durationSec, regenerated, skipped };
}

// ─── 场景插画 prompt 派生（纯本地，无 API）────────────────────────
function buildScenePrompt(meta: {
  scene_en: string;
  task_en: string;
  context: string;
}): string {
  // 从 context 截取第一句（中文标点之前）作为场景情境 hint
  const situation = meta.context.split(/[，。]/)[0].trim();
  return `A bright, minimalist flat illustration of ${meta.scene_en} scene, ${meta.task_en} task. ` +
    `Scene situation: ${situation}. ` +
    `Soft pastel colors (mint green, peach, light yellow). ` +
    `White background, no text, no real people faces, no watermark, ` +
    `9:16 portrait composition, vector art style, cheerful mood.`;
}

// ─── 场景插画生成 ────────────────────────────────────────────
async function generateSceneImage(
  descId: string,
  meta: any
): Promise<{ prompt: string; local_path: string; regenerated: boolean; skipped: boolean }> {
  const prompt = buildScenePrompt(meta);
  const outPath = join(IMAGE_BASE, `${descId}.jpg`);

  mkdirSync(IMAGE_BASE, { recursive: true });

  if (!FORCE && existsSync(outPath)) {
    console.log(`   ↻ 场景图已存在：${outPath}`);
    return { prompt, local_path: outPath, regenerated: false, skipped: true };
  }

  console.log(`   🎨 prompt: ${prompt}`);
  console.log(`   🎨 调用 text_to_image（base64 模式，避开签名 URL 403）...`);
  const imgs = await generate({
    prompt,
    aspectRatio: '9:16',
    model: 'image-01',
    responseFormat: 'base64',
  });
  const img = imgs[0];
  console.log(`      ✓ base64 length: ${img.base64?.length ?? 0} chars`);
  writeBase64ToFile(img, outPath);
  console.log(`      ✓ ${outPath}`);
  return { prompt, local_path: outPath, regenerated: true, skipped: false };
}

// ─── 主流程 ──────────────────────────────────────────────────
async function main() {
  if (!process.env.MINIMAX_API_KEY) {
    console.error('❌ MINIMAX_API_KEY 未设置');
    process.exit(1);
  }

  const descPath = process.argv[2] || DEFAULT_DESC;
  console.log('='.repeat(60));
  console.log(`🎬 全量资产生成 · ${descPath}`);
  console.log(`   mode: ${FORCE ? 'FORCE（重新生成全部）' : 'incremental（跳过已存在）'}`);
  console.log('='.repeat(60));

  const desc = JSON.parse(readFileSync(descPath, 'utf-8'));
  console.log(`\n📂 desc id: ${desc.id} · ${desc.cards.length} 张卡片`);

  // 1) 音频：每张卡片的每个 segment
  console.log('\n🎤 音频生成');
  let totalRegenerated = 0;
  let totalSkipped = 0;
  for (const card of desc.cards) {
    console.log(`\n📌 card ${card.index} (${card.type})`);
    const { durationSec, regenerated, skipped } = await generateCardAudio(desc.id, card);
    totalRegenerated += regenerated;
    totalSkipped += skipped;
    console.log(`   → duration_sec = ${durationSec}`);
  }

  // 2) 场景插画
  console.log('\n🎨 场景插画生成');
  const sceneResult = await generateSceneImage(desc.id, desc.meta);
  desc.scene_image = desc.scene_image || {};
  desc.scene_image.prompt = sceneResult.prompt;
  desc.scene_image.url = null;
  desc.scene_image.local_path = sceneResult.local_path;

  // 3) 重算总时长
  desc.duration_sec = desc.cards.reduce((s: number, c: any) => s + c.duration_sec, 0);
  desc.duration_frames = desc.duration_sec * desc.fps;

  // 4) 回写 desc
  writeFileSync(descPath, JSON.stringify(desc, null, 2) + '\n');

  console.log('\n' + '='.repeat(60));
  console.log('✅ 全部完成');
  console.log(`   音频：${totalRegenerated} 重新生成，${totalSkipped} 跳过`);
  console.log(`   插画：${sceneResult.regenerated ? '重新生成' : sceneResult.skipped ? '已存在跳过' : '未知'}`);
  console.log(`   总时长：${desc.duration_sec}s · ${desc.duration_frames} 帧 @ ${desc.fps}fps`);
  console.log(`   写回：${descPath}`);
  console.log('='.repeat(60));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});