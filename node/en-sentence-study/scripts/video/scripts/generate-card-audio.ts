// 根据 desc JSON 调用 TTS 生成指定卡片的音频，回填 duration_sec
// 用法：
//   pnpm exec tsx scripts/generate-card-audio.ts                          # 默认 desc/1.draft.json，所有卡片
//   pnpm exec tsx scripts/generate-card-audio.ts path/to/desc.json 1     # 指定 desc + 单张 card
//
// duration_sec 计算公式：max(4, ceil(sum_seg_ms / 1000) + 1)
//   至少 4 秒；超过则按总音频时长向上取整 + 1 秒留给动画
//
// 输出：
//   - 音频文件：public/audio/<desc-id>/<cardIdx>-<segIdx>-<lang>.mp3
//   - 更新 desc JSON：填充 tts_segments[].audio_path/duration_ms，回填 cards[].duration_sec
//                    重算总 duration_sec/duration_frames

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { synthesize } from '../src/api/minimax-tts';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

const DEFAULT_DESC = resolve(ROOT, 'scripts/desc/1.draft.json');
const AUDIO_DIR = resolve(ROOT, 'public/audio');

const MIN_DURATION_SEC = 4;
const PADDING_SEC = 1;
const PAUSE_MS = 700; // 段间停顿（与 src/theme.ts 同步）

async function generateForCard(
  desc: any,
  card: any,
  audioDir: string
): Promise<{ totalMs: number; durationSec: number }> {
  const cardAudioDir = join(audioDir, desc.id);
  mkdirSync(cardAudioDir, { recursive: true });

  let totalMs = 0;
  const segments: any[] = [];

  for (let i = 0; i < card.tts_segments.length; i++) {
    const seg = card.tts_segments[i];
    const outName = `${card.index}-${i}-${seg.lang}.mp3`;
    const outPath = join(cardAudioDir, outName);

    // 已存在则跳过（方便重跑）
    if (existsSync(outPath)) {
      console.log(`   ↻ 已存在 ${outName}，跳过`);
      // 从文件名读 ms 需要 ffprobe，这里简化：标记为 0 让 totalMs 由外部补
      segments.push({ ...seg, audio_path: outPath, duration_ms: 0, skipped: true });
      continue;
    }

    console.log(`   🎤 card ${card.index} seg ${i} (${seg.lang}): "${seg.text}"`);
    const { buffer, durationMs } = await synthesize({
      text: seg.text,
      language: seg.lang,
    });
    writeFileSync(outPath, buffer);
    totalMs += durationMs;
    segments.push({ ...seg, audio_path: outPath, duration_ms: durationMs });
    console.log(`      ✓ ${(buffer.length / 1024).toFixed(1)} KB · ${(durationMs / 1000).toFixed(2)}s`);
  }

  // 对跳过的 segment 用 ffprobe 拿真实时长（如果系统装了）
  for (const seg of segments) {
    if (seg.skipped) {
      try {
        const { execSync } = await import('node:child_process');
        const out = execSync(
          `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${seg.audio_path}"`,
          { encoding: 'utf-8' }
        );
        seg.duration_ms = Math.round(parseFloat(out.trim()) * 1000);
        totalMs += seg.duration_ms;
        console.log(`      ↻ 实际时长 ${(seg.duration_ms / 1000).toFixed(2)}s`);
      } catch {
        // 没装 ffprobe 时给个保守估算（按 4 字/秒）
        seg.duration_ms = seg.text.length * 250;
        totalMs += seg.duration_ms;
        console.log(`      ↻ 估算时长 ${(seg.duration_ms / 1000).toFixed(2)}s（无 ffprobe）`);
      }
    }
  }

  // 包含段间停顿：n 段 → n-1 个停顿
  const pauseTotalMs = Math.max(0, segments.length - 1) * PAUSE_MS;
  const durationSec = Math.max(
    MIN_DURATION_SEC,
    Math.ceil((totalMs + pauseTotalMs) / 1000) + PADDING_SEC
  );

  // 回写到 card.tts_segments
  card.tts_segments = segments.map((s) => ({
    lang: s.lang,
    text: s.text,
    audio_path: s.audio_path,
    duration_ms: s.duration_ms,
  }));

  return { totalMs, durationSec };
}

async function main() {
  if (!process.env.MINIMAX_API_KEY) {
    console.error('❌ MINIMAX_API_KEY 未设置');
    process.exit(1);
  }

  const descPath = process.argv[2] || DEFAULT_DESC;
  const cardArg = process.argv[3]; // '1' / '2' / 'all' / 不传
  const targetAll = !cardArg || cardArg === 'all';

  console.log(`📂 desc: ${descPath}`);
  console.log(`🎯 target: ${targetAll ? 'all cards' : `card index ${cardArg}`}`);

  const desc = JSON.parse(readFileSync(descPath, 'utf-8'));

  const targets = targetAll
    ? desc.cards
    : desc.cards.filter((c: any) => c.index === parseInt(cardArg, 10));

  if (targets.length === 0) {
    console.error(`❌ 没找到匹配的 card`);
    process.exit(1);
  }

  for (const card of targets) {
    console.log(`\n📌 card ${card.index} (${card.type})`);
    const { durationSec } = await generateForCard(desc, card, AUDIO_DIR);
    card.duration_sec = durationSec;
    console.log(`   → duration_sec = ${durationSec}`);
  }

  // 重算总时长
  if (targetAll) {
    desc.duration_sec = desc.cards.reduce((s: number, c: any) => s + c.duration_sec, 0);
    desc.duration_frames = desc.duration_sec * desc.fps;
    console.log(`\n📐 总时长: ${desc.duration_sec}s · ${desc.duration_frames} 帧`);
  }

  writeFileSync(descPath, JSON.stringify(desc, null, 2) + '\n');
  console.log(`\n✅ 已更新 ${descPath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});