// scripts/video/scripts/studio.ts
// 启动 Remotion Studio 预览指定 desc JSON
//
// 用法：
//   pnpm exec tsx scripts/studio.ts <desc.json>
//
// 例：
//   pnpm exec tsx scripts/studio.ts scripts/desc/1.draft.json
//   pnpm exec tsx scripts/studio.ts scripts/output/1.json   # 也能跑，但 props 直接吃 desc JSON
//
// 与 render.ts 的 phaseVideo 行为对齐：把 desc JSON 包成 { desc } 传入
// EnSentenceVideo 这个 Composition（Root.tsx 已配置 calculateMetadata
// 从 props.desc 动态算 fps / durationInFrames）。

import { readFileSync, existsSync } from 'node:fs';
import { dirname, resolve, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');              // scripts/video/
const CWD = process.cwd();                          // 用户调用时的当前目录

function printHelp() {
  console.log(`
scripts/studio.ts - 启动 Remotion Studio 预览指定 desc

用法：
  pnpm exec tsx scripts/studio.ts <desc.json>

示例：
  pnpm exec tsx scripts/studio.ts scripts/desc/1.draft.json
  pnpm exec tsx scripts/studio.ts scripts/desc/3.draft.json
`);
}

function main() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args[0] === '-h' || args[0] === '--help') {
    printHelp();
    process.exit(args.length === 0 ? 1 : 0);
  }

  // 路径基准跟 render.ts 的 phaseDesc / phaseVideo 对齐：
  //   - 绝对路径直接用
  //   - 相对路径基于 process.cwd()（用户在 scripts/video/ 里跑，
  //     写 scripts/desc/8.draft.json 就是 scripts/video/scripts/desc/...）
  // 这样 `render.ts` 和 `studio.ts` 行为一致。
  const inputArg = args[0];
  const descPath = resolve(CWD, inputArg);

  if (!existsSync(descPath)) {
    console.error(`❌ desc 文件不存在：${descPath}`);
    console.error(`   当前 cwd：${CWD}`);
    console.error(`   提示：相对路径基于当前目录；要么 cd 到 scripts/video/ 后用`);
    console.error(`         scripts/desc/8.draft.json，要么用绝对路径。`);
    process.exit(1);
  }

  // 读 desc，跟 phaseVideo 一样包成 { desc } 喂给 EnSentenceVideo
  const desc = JSON.parse(readFileSync(descPath, 'utf-8'));
  const propsJson = JSON.stringify({ desc });
  const id = basename(descPath, '.json');

  console.log('='.repeat(60));
  console.log(`🎬 Remotion Studio · ${id}`);
  console.log(`   desc:  ${descPath}`);
  console.log(`   props: ${propsJson.slice(0, 80)}${propsJson.length > 80 ? '...' : ''}`);
  console.log('='.repeat(60));

  const r = spawnSync(
    'pnpm',
    [
      'exec', 'remotion', 'studio',
      'src/index.ts',
      '--props', propsJson,
    ],
    { cwd: ROOT, stdio: 'inherit' }
  );

  if (r.status !== 0) process.exit(r.status ?? 1);
}

main();