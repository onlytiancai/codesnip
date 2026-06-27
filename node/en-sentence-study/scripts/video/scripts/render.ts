// scripts/video/scripts/render.ts
// 端到端视频生成流水线（Step 10）：支持分阶段执行 + 渲染后清理中间产物
//
// 用法：
//   pnpm exec tsx scripts/render.ts <input.json> [options]
//
// 4 个阶段（可单独或全部执行）：
//   desc    把 scripts/output/N.json 解析成 scripts/desc/N.draft.json（纯本地）
//   assets  调用 MiniMax text_to_image 生成场景插图 public/images/N.jpg
//   audio   调用 MiniMax TTS 生成所有卡片音频 public/audio/N/*.mp3
//   video   用 Remotion 渲染最终视频 out/N-desc.mp4
//
// 选项：
//   -p, --phase <name>    指定阶段（desc|assets|audio|video|all），默认 all
//   -f, --force           强制重新生成，跳过已存在检测
//   -c, --clean           渲染完成后清理中间产物（图片/音频/desc）
//       --keep-images     --clean 时保留图片
//       --keep-audio      --clean 时保留音频
//       --keep-desc       --clean 时保留 desc JSON
//   -i, --input <path>    源 JSON（也可用位置参数）
//   -d, --desc <path>     直接指定 desc JSON 跳过 desc 阶段
//   -o, --output <path>   输出 mp4 路径（默认 out/<id>-desc.mp4）
//   -h, --help            显示帮助

import { readFileSync, writeFileSync, mkdirSync, existsSync, rmSync } from 'node:fs';
import { dirname, resolve, basename, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execSync, spawnSync } from 'node:child_process';

import { synthesize } from '../src/api/minimax-tts';
import { generate, writeBase64ToFile } from '../src/api/minimax-image';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');              // scripts/video/
const PROJECT_ROOT = resolve(ROOT, '../..');        // en-sentence-study/（用户给 scripts/output/N.json 的基准）

// ─── 共享常量（与 generate-all.ts / theme.ts 同步）────────────
const MIN_DURATION_SEC = 4;
const PADDING_SEC = 1;
const PAUSE_MS = 700;

type Phase = 'desc' | 'assets' | 'audio' | 'video' | 'all';

type CliOptions = {
  input?: string;
  desc?: string;
  phase: Phase;
  force: boolean;
  clean: boolean;
  keepImages: boolean;
  keepAudio: boolean;
  keepDesc: boolean;
  output?: string;
};

// ─── CLI 解析 ─────────────────────────────────────────────
function parseArgs(argv: string[]): CliOptions {
  const args = argv.slice(2);
  const opts: CliOptions = {
    phase: 'all',
    force: false,
    clean: false,
    keepImages: false,
    keepAudio: false,
    keepDesc: false,
  };

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    const take = () => args[++i];
    if (a === '--phase' || a === '-p') opts.phase = take() as Phase;
    else if (a === '--force' || a === '-f') opts.force = true;
    else if (a === '--clean' || a === '-c') opts.clean = true;
    else if (a === '--keep-images') opts.keepImages = true;
    else if (a === '--keep-audio') opts.keepAudio = true;
    else if (a === '--keep-desc') opts.keepDesc = true;
    else if (a === '--input' || a === '-i') opts.input = take();
    else if (a === '--desc' || a === '-d') opts.desc = take();
    else if (a === '--output' || a === '-o') opts.output = take();
    else if (a === '--help' || a === '-h') {
      printHelp();
      process.exit(0);
    } else if (!a.startsWith('-') && !opts.input) {
      opts.input = a;
    } else {
      console.error(`❌ 未知参数：${a}`);
      printHelp();
      process.exit(1);
    }
  }

  if (!['desc', 'assets', 'audio', 'video', 'all'].includes(opts.phase)) {
    console.error(`❌ 未知 phase：${opts.phase}（应为 desc|assets|audio|video|all）`);
    process.exit(1);
  }
  return opts;
}

function printHelp() {
  console.log(`
scripts/render.ts - 端到端视频生成流水线

用法：
  pnpm exec tsx scripts/render.ts <input.json> [options]

阶段（--phase）：
  desc    生成 scripts/desc/<id>.draft.json
  assets  生成场景插图 public/images/<id>.jpg
  audio   生成所有卡片 TTS 音频 public/audio/<id>/*.mp3
  video   用 Remotion 渲染 out/<id>-desc.mp4
  all     顺序执行 4 个阶段（默认）

选项：
  -p, --phase <name>    指定阶段（默认 all）
  -f, --force           强制重新生成，跳过已存在的检测
  -c, --clean           渲染完成后清理中间产物（图片/音频/desc）
      --keep-images     --clean 时保留图片
      --keep-audio      --clean 时保留音频
      --keep-desc       --clean 时保留 desc JSON
  -i, --input <path>    源 JSON（也可用位置参数）
  -d, --desc <path>     直接指定 desc JSON 跳过 desc 阶段
  -o, --output <path>   输出 mp4 路径（默认 out/<id>-desc.mp4）
  -h, --help            显示帮助

示例：
  # 端到端跑 1.json
  pnpm exec tsx scripts/render.ts scripts/output/1.json

  # 强制重新生成所有 + 渲染后清理中间产物（只留 mp4）
  pnpm exec tsx scripts/render.ts scripts/output/1.json --force --clean

  # 只生成 desc JSON（手动审核后再渲染）
  pnpm exec tsx scripts/render.ts scripts/output/1.json --phase desc

  # 渲染指定 desc（跳过前面阶段）
  pnpm exec tsx scripts/render.ts --phase video --desc scripts/desc/1.draft.json

  # 渲染后清理（保留图片）
  pnpm exec tsx scripts/render.ts --phase video --desc scripts/desc/1.draft.json --clean --keep-images
`);
}

// ─── 阶段 1：生成 desc JSON（spawn 到现有 generate-desc.ts）──────
function phaseDesc(inputPath: string, force: boolean): string {
  const id = basename(inputPath, '.json');
  const outPath = resolve(ROOT, 'scripts/desc', `${id}.draft.json`);

  if (!force && existsSync(outPath)) {
    console.log(`↻ desc 已存在，跳过：${outPath}`);
    return outPath;
  }

  console.log(`📝 阶段 desc · 源 ${inputPath}`);
  const r = spawnSync('pnpm', ['exec', 'tsx', 'scripts/generate-desc.ts', inputPath], {
    cwd: ROOT,
    stdio: 'inherit',
  });
  if (r.status !== 0) throw new Error('desc 阶段失败');
  return outPath;
}

// ─── 阶段 2：生成场景插图 ────────────────────────────────────────
function buildScenePrompt(meta: {
  scene_en: string;
  task_en: string;
  context: string;
}): string {
  const situation = meta.context.split(/[，。]/)[0].trim();
  return `A bright, minimalist flat illustration of ${meta.scene_en} scene, ${meta.task_en} task. ` +
    `Scene situation: ${situation}. ` +
    `Soft pastel colors (mint green, peach, light yellow). ` +
    `White background, no text, no real people faces, no watermark, ` +
    `16:9 landscape composition, vector art style, cheerful mood.`;
}

async function phaseAssets(descPath: string, force: boolean): Promise<void> {
  const desc = JSON.parse(readFileSync(descPath, 'utf-8'));
  const id = desc.id as string;
  const outPath = resolve(ROOT, 'public/images', `${id}.jpg`);
  mkdirSync(resolve(ROOT, 'public/images'), { recursive: true });

  if (!force && existsSync(outPath)) {
    console.log(`↻ 插图已存在，跳过：${outPath}`);
    return;
  }

  const prompt = buildScenePrompt(desc.meta);
  console.log(`🎨 阶段 assets · ${id}`);
  console.log(`   prompt: ${prompt.slice(0, 80)}...`);
  const imgs = await generate({
    prompt,
    aspectRatio: '16:9',
    model: 'image-01',
    responseFormat: 'base64',
  });
  writeBase64ToFile(imgs[0], outPath);
  console.log(`   ✓ ${outPath}`);

  // 回填到 desc（只更新 scene_image 字段，不动其它）
  desc.scene_image = desc.scene_image || {};
  desc.scene_image.prompt = prompt;
  desc.scene_image.url = null;
  desc.scene_image.local_path = outPath;
  writeFileSync(descPath, JSON.stringify(desc, null, 2) + '\n');
}

// ─── 阶段 3：生成所有卡片音频 ─────────────────────────────────────
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

async function phaseAudio(descPath: string, force: boolean): Promise<void> {
  const desc = JSON.parse(readFileSync(descPath, 'utf-8'));
  const audioBaseDir = resolve(ROOT, 'public/audio', desc.id);
  mkdirSync(audioBaseDir, { recursive: true });

  console.log(`🎤 阶段 audio · ${desc.id}（${desc.cards.length} 张卡片）`);
  for (const card of desc.cards) {
    console.log(`\n   📌 card ${card.index} (${card.type})`);
    let totalMs = 0;

    for (let i = 0; i < card.tts_segments.length; i++) {
      const seg = card.tts_segments[i];
      const outName = `${card.index}-${i}-${seg.lang}.mp3`;
      const outPath = join(audioBaseDir, outName);

      if (!force && existsSync(outPath)) {
        const ms = measureMs(outPath, seg.text.length);
        seg.audio_path = outPath;
        seg.duration_ms = ms;
        totalMs += ms;
        console.log(`      ↻ 已存在 ${outName} · ${(ms / 1000).toFixed(2)}s`);
        continue;
      }

      console.log(`      🎤 seg ${i} (${seg.lang}): "${seg.text}"`);
      const { buffer, durationMs } = await synthesize({
        text: seg.text,
        language: seg.lang,
      });
      writeFileSync(outPath, buffer);
      seg.audio_path = outPath;
      seg.duration_ms = durationMs;
      totalMs += durationMs;
      console.log(`         ✓ ${(buffer.length / 1024).toFixed(1)} KB · ${(durationMs / 1000).toFixed(2)}s`);
    }

    const pauseTotalMs = Math.max(0, card.tts_segments.length - 1) * PAUSE_MS;
    const durationSec = Math.max(
      MIN_DURATION_SEC,
      Math.ceil((totalMs + pauseTotalMs) / 1000) + PADDING_SEC
    );
    card.duration_sec = durationSec;
    console.log(`      → duration_sec = ${durationSec}`);
  }

  desc.duration_sec = desc.cards.reduce((s: number, c: any) => s + c.duration_sec, 0);
  desc.duration_frames = desc.duration_sec * desc.fps;
  console.log(`\n   📐 总时长：${desc.duration_sec}s · ${desc.duration_frames} 帧 @ ${desc.fps}fps`);
  writeFileSync(descPath, JSON.stringify(desc, null, 2) + '\n');
}

// ─── 阶段 4：用 Remotion 渲染视频 ─────────────────────────────────
function phaseVideo(descPath: string, outputPath: string): void {
  console.log(`🎬 阶段 video · ${descPath} → ${outputPath}`);
  const desc = JSON.parse(readFileSync(descPath, 'utf-8'));
  const propsJson = JSON.stringify({ desc });

  const r = spawnSync(
    'pnpm',
    [
      'exec', 'remotion', 'render',
      'src/index.ts', 'EnSentenceVideo', outputPath,
      '--props', propsJson,
    ],
    { cwd: ROOT, stdio: 'inherit' }
  );
  if (r.status !== 0) throw new Error('video 渲染失败');
}

// ─── 清理中间产物（仅 video/all 阶段之后生效）─────────────────────
function phaseClean(opts: {
  id: string;
  descPath: string;
  keepImages: boolean;
  keepAudio: boolean;
  keepDesc: boolean;
}): { removed: string[] } {
  const removed: string[] = [];

  if (!opts.keepImages) {
    const p = resolve(ROOT, 'public/images', `${opts.id}.jpg`);
    if (existsSync(p)) {
      rmSync(p);
      removed.push(p);
    }
  }
  if (!opts.keepAudio) {
    const p = resolve(ROOT, 'public/audio', opts.id);
    if (existsSync(p)) {
      rmSync(p, { recursive: true });
      removed.push(p);
    }
  }
  if (!opts.keepDesc) {
    if (existsSync(opts.descPath)) {
      rmSync(opts.descPath);
      removed.push(opts.descPath);
    }
  }
  return { removed };
}

// ─── 主流程 ───────────────────────────────────────────────────
async function main() {
  const opts = parseArgs(process.argv);

  // 解析 input / desc 路径
  //   input 基于 PROJECT_ROOT（项目根）—— 用户习惯给 scripts/output/N.json
  //   desc 基于 ROOT（video 目录）—— 默认在 scripts/desc/ 下
  const inputPath = opts.input ? resolve(PROJECT_ROOT, opts.input) : undefined;
  const descPath = opts.desc ? resolve(opts.desc) : undefined;

  if (!inputPath && !descPath && opts.phase !== 'video') {
    console.error('❌ 必须指定 --input 或 --desc（--phase=video 也需要 --desc）');
    printHelp();
    process.exit(1);
  }

  // 阶段参数 + 必要依赖校验
  if ((opts.phase === 'desc' || opts.phase === 'all') && !inputPath) {
    console.error('❌ --phase=desc 需要 --input');
    process.exit(1);
  }

  // 检查 API key（任何需要调 API 的阶段都要）
  const needApi = ['assets', 'audio', 'all'].includes(opts.phase);
  if (needApi && !process.env.MINIMAX_API_KEY) {
    console.error('❌ MINIMAX_API_KEY 未设置（assets/audio 阶段需要）');
    process.exit(1);
  }

  console.log('='.repeat(60));
  console.log(`🎬 视频生成流水线`);
  console.log(`   phase: ${opts.phase}${opts.force ? ' · FORCE' : ''}${opts.clean ? ' · CLEAN' : ''}`);
  if (inputPath) console.log(`   input: ${inputPath}`);
  if (descPath) console.log(`   desc:  ${descPath}`);
  console.log('='.repeat(60));

  let finalDescPath: string | undefined;

  // 从 input 自动推导 desc 路径（如果用户没显式给 --desc）
  const inferredDescPath = inputPath
    ? resolve(ROOT, 'scripts/desc', `${basename(inputPath, '.json')}.draft.json`)
    : undefined;

  // 顺序执行各阶段
  if (opts.phase === 'desc' || opts.phase === 'all') {
    finalDescPath = phaseDesc(inputPath!, opts.force);
    console.log(`   ✓ desc → ${finalDescPath}`);
  } else if (inferredDescPath && existsSync(inferredDescPath)) {
    // 非 desc 阶段：若 input 对应的 desc 已存在，自动复用（无需 --desc）
    finalDescPath = inferredDescPath;
  }
  if (opts.phase === 'assets' || opts.phase === 'all') {
    finalDescPath = finalDescPath || descPath!;
    if (!existsSync(finalDescPath)) throw new Error(`desc 不存在：${finalDescPath}`);
    await phaseAssets(finalDescPath, opts.force);
    console.log(`   ✓ assets`);
  }
  if (opts.phase === 'audio' || opts.phase === 'all') {
    finalDescPath = finalDescPath || descPath!;
    if (!existsSync(finalDescPath)) throw new Error(`desc 不存在：${finalDescPath}`);
    await phaseAudio(finalDescPath, opts.force);
    console.log(`   ✓ audio`);
  }

  let outputPath: string | undefined;
  if (opts.phase === 'video' || opts.phase === 'all') {
    finalDescPath = finalDescPath || descPath!;
    if (!existsSync(finalDescPath)) throw new Error(`desc 不存在：${finalDescPath}`);
    const desc = JSON.parse(readFileSync(finalDescPath, 'utf-8'));
    outputPath = opts.output
      ? resolve(opts.output)
      : resolve(ROOT, 'out', `${desc.id}-desc.mp4`);
    mkdirSync(dirname(outputPath), { recursive: true });
    phaseVideo(finalDescPath, outputPath);
    console.log(`   ✓ video → ${outputPath}`);
  }

  // clean 阶段
  if (opts.clean && finalDescPath && outputPath) {
    const desc = JSON.parse(readFileSync(finalDescPath, 'utf-8'));
    const { removed } = phaseClean({
      id: desc.id,
      descPath: finalDescPath,
      keepImages: opts.keepImages,
      keepAudio: opts.keepAudio,
      keepDesc: opts.keepDesc,
    });
    console.log('');
    if (removed.length) {
      console.log(`🧹 已清理 ${removed.length} 项中间产物：`);
      for (const p of removed) console.log(`   - ${p}`);
    }
    console.log(`✅ 仅保留最终视频：${outputPath}`);
  }

  console.log('='.repeat(60));
  console.log('🎉 完成');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});