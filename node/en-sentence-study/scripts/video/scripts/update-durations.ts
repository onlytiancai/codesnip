// 重算 desc JSON 的 duration_sec：含段间停顿（zh→PAUSE→en）
// 用法：
//   pnpm exec tsx scripts/update-durations.ts                          # 默认 desc/1.draft.json
//   pnpm exec tsx scripts/update-durations.ts path/to/desc.json
//
// 公式：duration_sec = max(4, ceil((sum_seg_ms + (n-1) * PAUSE_MS) / 1000) + 1)
//   至少 4 秒；按总时长向上取整 + 1 秒留给淡出动画
//   段间停顿 PAUSE_MS 与 src/theme.ts 保持一致

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

const DEFAULT_DESC = resolve(ROOT, 'scripts/desc/1.draft.json');
const PAUSE_MS = 700;          // 与 src/theme.ts 同步
const MIN_DURATION_SEC = 4;
const PADDING_SEC = 1;

function calcDuration(segMs: number[]): number {
  const sumMs = segMs.reduce((s, m) => s + m, 0);
  const pauseTotalMs = (segMs.length - 1) * PAUSE_MS;
  return Math.max(MIN_DURATION_SEC, Math.ceil((sumMs + pauseTotalMs) / 1000) + PADDING_SEC);
}

function main() {
  const descPath = process.argv[2] || DEFAULT_DESC;
  const desc = JSON.parse(readFileSync(descPath, 'utf-8'));

  console.log(`📂 ${descPath}`);
  console.log(`   PAUSE_MS = ${PAUSE_MS} (段间停顿)\n`);

  for (const card of desc.cards) {
    const segMs = card.tts_segments.map((s: any) => s.duration_ms);
    const oldSec = card.duration_sec;
    card.duration_sec = calcDuration(segMs);
    const arrow = card.duration_sec !== oldSec ? `${oldSec} → ${card.duration_sec}` : `${card.duration_sec} (不变)`;
    console.log(`   card ${card.index} (${card.type}): ${arrow}s  · ${segMs.length} 段`);
  }

  desc.duration_sec = desc.cards.reduce((s: number, c: any) => s + c.duration_sec, 0);
  desc.duration_frames = desc.duration_sec * desc.fps;

  writeFileSync(descPath, JSON.stringify(desc, null, 2) + '\n');

  console.log(`\n📐 总时长：${desc.duration_sec}s · ${desc.duration_frames} 帧`);
  console.log(`✅ 已写回 ${descPath}`);
}

main();