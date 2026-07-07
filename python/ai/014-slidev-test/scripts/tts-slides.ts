// CLI：从 slides.md 抽取 v-click 文本，逐段合成语音，写入 public/audio/。
//
// 用法：
//   pnpm tts:slides                      默认生成所有缺失片段
//   pnpm tts:slides -- --dry             只打印计划，不打 API（验证文本提取）
//   pnpm tts:slides -- --force           覆盖已存在的 mp3
//   pnpm tts:slides -- --slide 1 --click 1   只生成指定单段
//
// 网络失败兜底：若 fetch 报 ECONNREFUSED/超时，可临时设
//   HTTPS_PROXY=http://127.0.0.1:10808 pnpm tts:slides

import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { extractClicksFromFile, type ClickSegment } from './extract-clicks.ts';
import { synthesize } from './minimax-tts.ts';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const SLIDES = resolve(ROOT, 'slides.md');
const AUDIO_DIR = resolve(ROOT, 'public/audio');

type Args = {
  dry: boolean;
  force: boolean;
  slide?: number;
  click?: number;
};

function parseArgs(argv: string[]): Args {
  const args: Args = { dry: false, force: false };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--dry') args.dry = true;
    else if (a === '--force') args.force = true;
    else if (a === '--slide') args.slide = Number(argv[++i]);
    else if (a === '--click') args.click = Number(argv[++i]);
  }
  return args;
}

function audioName(seg: ClickSegment): string {
  return `slide-${seg.slide}-click-${seg.click}.mp3`;
}

/**
 * 写 `public/audio/manifest.json`：浏览器里的 global-bottom.vue 读这个文件
 * 拿到 (slide, click) → audio 映射，不再硬编码在 Vue 组件里。
 *
 * 字段约定：
 *   - slide / click：与 slides.md 中的 narrate N 对齐
 *   - audio：浏览器可访问的 URL 路径（以 / 开头，public/ 下文件原样发布到站点根）
 *   - text：原文（仅供调试，浏览器不读）
 */
function writeManifest(segments: ClickSegment[]): void {
  const items = segments.map((seg) => ({
    slide: seg.slide,
    click: seg.click,
    audio: `/audio/${audioName(seg)}`,
    text: seg.text,
  }));
  const manifestPath = resolve(AUDIO_DIR, 'manifest.json');
  mkdirSync(AUDIO_DIR, { recursive: true });
  writeFileSync(manifestPath, JSON.stringify({ items }, null, 2) + '\n');
  console.log(`清单已写入 ${manifestPath}（${items.length} 条）`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  let segments = extractClicksFromFile(SLIDES);
  if (args.slide != null) {
    segments = segments.filter((s) => s.slide === args.slide);
  }
  if (args.click != null) {
    segments = segments.filter((s) => s.click === args.click);
  }

  if (segments.length === 0) {
    console.log('没有匹配的 v-click 段落。');
    return;
  }

  console.log(`抽取到 ${segments.length} 段 v-click 旁白：\n`);
  for (const seg of segments) {
    console.log(`  [slide ${seg.slide} click ${seg.click}] ${seg.text}`);
  }
  console.log('');

  if (args.dry) {
    console.log('--dry：仅打印计划，不调用 API。');
    // 即使 dry 也写一份 manifest，方便 dev 时预览旁白清单
    writeManifest(segments);
    return;
  }

  mkdirSync(AUDIO_DIR, { recursive: true });

  for (const seg of segments) {
    const name = audioName(seg);
    const outPath = resolve(AUDIO_DIR, name);

    if (existsSync(outPath) && !args.force) {
      console.log(`跳过 ${name}（已存在，用 --force 覆盖）`);
      continue;
    }

    const t0 = Date.now();
    process.stdout.write(`合成 ${name} ... `);
    try {
      const { buffer, durationMs } = await synthesize({
        text: seg.text,
        language: 'zh',
        voiceId: 'female-shaonv',
      });
      writeFileSync(outPath, buffer);
      const kb = (buffer.length / 1024).toFixed(1);
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      console.log(`OK  ${kb}KB  音频${(durationMs / 1000).toFixed(1)}s  耗时${elapsed}s`);
    } catch (err) {
      console.log('失败');
      console.error(`  ${(err as Error).message}`);
      if (/ECONNREFUSED|ETIMEDOUT|fetch failed/i.test((err as Error).message)) {
        console.error('  提示：网络失败可试 HTTPS_PROXY=http://127.0.0.1:10808 pnpm tts:slides');
      }
      process.exitCode = 1;
    }
  }

  // 始终重写 manifest：保证和 slides.md 当前抽取结果一致（含被跳过的旧 mp3）
  writeManifest(segments);

  console.log(`\n完成。输出目录：${AUDIO_DIR}`);
}

main();
