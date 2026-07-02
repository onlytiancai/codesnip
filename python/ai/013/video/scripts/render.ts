/**
 * XOR 反向传播视频 · 端到端 pipeline (Remotion 端)
 *
 * --phase assets  拷贝 pipeline/output/{diagrams,plots,formulas}/*.png → public/images/
 * --phase video   渲染 out/xor-bp.mp4
 *
 * TTS 阶段由 Python pipeline/render_audio.py 处理。
 *
 * 用法:
 *   pnpm exec tsx scripts/render.ts --phase assets
 *   pnpm exec tsx scripts/render.ts --phase video
 *   pnpm exec tsx scripts/render.ts --phase all
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, rmSync } from 'node:fs';
import { dirname, resolve, basename, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync, execSync } from 'node:child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');                          // video/
const PROJECT_ROOT = resolve(ROOT, '..');                       // /013/
const PYTHON = '/Users/huhao/.pyenv/versions/3.11.9/bin/python';

type Phase = 'assets' | 'video' | 'all';

type CliOptions = {
  phase: Phase;
  force: boolean;
  output?: string;
};

function parseArgs(argv: string[]): CliOptions {
  const args = argv.slice(2);
  const opts: CliOptions = { phase: 'all', force: false };

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    const take = () => args[++i];
    if (a === '--phase' || a === '-p') opts.phase = take() as Phase;
    else if (a === '--force' || a === '-f') opts.force = true;
    else if (a === '--output' || a === '-o') opts.output = take();
    else if (a === '--help' || a === '-h') {
      printHelp();
      process.exit(0);
    } else {
      console.error(`❌ 未知参数: ${a}`);
      process.exit(1);
    }
  }
  if (!['assets', 'video', 'all'].includes(opts.phase)) {
    console.error(`❌ phase 必须是 assets|video|all`);
    process.exit(1);
  }
  return opts;
}

function printHelp() {
  console.log(`
scripts/render.ts - XOR 视频端到端 pipeline

用法:
  pnpm exec tsx scripts/render.ts [--phase assets|video|all] [--force] [--output out.mp4]

阶段:
  assets  拷贝 pipeline/output/{diagrams,plots,formulas}/*.png → public/images/
  video   调 Remotion 渲染最终 mp4
  all     两个都跑(默认)

示例:
  pnpm exec tsx scripts/render.ts --phase assets
  pnpm exec tsx scripts/render.ts --phase video
  pnpm exec tsx scripts/render.ts --phase all
`);
}

// ── 阶段 1: assets ────────────────────────────────
function phaseAssets(force: boolean): void {
  if (!existsSync(PYTHON)) {
    console.warn(`⚠️ 未找到 Python: ${PYTHON},跳过 assets 拷贝(请手动 cp)`);
    return;
  }
  console.log('🎨 阶段 assets · 调用 pipeline/render_assets.py');
  const r = spawnSync(PYTHON, ['pipeline/render_assets.py'], {
    cwd: PROJECT_ROOT,
    stdio: 'inherit',
    env: { ...process.env },
  });
  if (r.status !== 0) throw new Error('render_assets.py 失败');

  // 拷贝 PNG 到 public/images/
  const imagesDir = resolve(ROOT, 'public/images');
  mkdirSync(imagesDir, { recursive: true });
  const subs = ['diagrams', 'plots', 'formulas'];
  let copied = 0;
  for (const sub of subs) {
    const srcDir = resolve(PROJECT_ROOT, 'pipeline/output', sub);
    if (!existsSync(srcDir)) continue;
    const files = execSync(`ls ${srcDir}/*.png 2>/dev/null || true`,
                           { shell: '/bin/bash' })
                  .toString().trim().split('\n').filter(Boolean);
    for (const f of files) {
      const fname = basename(f);
      const dst = resolve(imagesDir, fname);
      if (!force && existsSync(dst)) continue;
      execSync(`cp "${f}" "${dst}"`);
      copied++;
    }
  }
  console.log(`   ✓ 拷贝 ${copied} 张 PNG 到 public/images/`);
}

// ── 阶段 2: video ────────────────────────────────
function phaseVideo(outputPath: string, force: boolean): void {
  const descPath = resolve(PROJECT_ROOT, 'desc.json');
  if (!existsSync(descPath)) {
    throw new Error(`desc.json 不存在: ${descPath}（请先跑 audio 阶段回填 duration）`);
  }
  const desc = JSON.parse(readFileSync(descPath, 'utf-8'));
  const propsJson = JSON.stringify({ desc });

  mkdirSync(dirname(outputPath), { recursive: true });
  console.log(`🎬 阶段 video · → ${outputPath}`);
  console.log(`   duration: ${desc.duration_sec}s · ${desc.duration_frames} frames`);

  const r = spawnSync(
    'pnpm',
    [
      'exec', 'remotion', 'render',
      'src/index.ts', 'XorBPVideo', outputPath,
      '--props', propsJson,
    ],
    { cwd: ROOT, stdio: 'inherit' }
  );
  if (r.status !== 0) throw new Error('remotion render 失败');
}

// ── main ────────────────────────────────
async function main() {
  const opts = parseArgs(process.argv);
  console.log('='.repeat(60));
  console.log(`🎬 XOR 视频 pipeline · phase=${opts.phase}${opts.force ? ' · FORCE' : ''}`);
  console.log('='.repeat(60));

  if (opts.phase === 'assets' || opts.phase === 'all') {
    phaseAssets(opts.force);
    console.log('   ✓ assets');
  }
  if (opts.phase === 'video' || opts.phase === 'all') {
    const out = opts.output ?? resolve(ROOT, 'out/xor-bp.mp4');
    phaseVideo(out, opts.force);
    console.log(`   ✓ video → ${out}`);
  }

  console.log('='.repeat(60));
  console.log('🎉 完成');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
