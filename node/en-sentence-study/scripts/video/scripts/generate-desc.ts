// 根据源 JSON 纯本地生成 desc JSON（不调用任何外部 API）
// 用法：
//   pnpm exec tsx scripts/generate-desc.ts                    # 默认 ../output/1.json
//   pnpm exec tsx scripts/generate-desc.ts path/to/N.json    # 指定源
//
// 输出：scripts/desc/<id>.draft.json
//
// 关键规则：
//   - duration_sec 一律填 -1（待音频生成后再回填）
//   - card 0 (intro)：tts_segments = [scene_zh + task_zh]，仅中文
//   - card 1..N (expression)：tts_segments = [literal_translation, sentence]，先 zh 后 en
//   - card N+1 (summary)：tts_segments = [explanation]，仅中文

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

// 解析脚本 video 目录的相对路径（ROOT = scripts/video，再往上一层到项目根）
const DEFAULT_SOURCE = resolve(ROOT, '../output/1.json');
const OUTPUT_DIR = resolve(ROOT, 'scripts/desc');

const FPS = 30;
const THEME: 'mint' | 'sunny' = 'mint';

type Translation = {
  sentence: string;
  style: 'polite' | 'neutral' | 'casual' | 'bold';
  note: string;
  literal_translation: string;
  phonetic: string;
};

type Source = {
  scene_en: string;
  scene_zh: string;
  task_en: string;
  task_zh: string;
  context: string;
  sentence_zh: string;
  translations: Translation[];
  explanation: string;
};

type Card = {
  index: number;
  type: 'intro' | 'expression' | 'summary';
  duration_sec: number; // -1 占位
  tts_segments: Array<{ lang: 'zh' | 'en'; text: string }>;
  style?: 'polite' | 'neutral' | 'casual' | 'bold';
  sentence_en?: string;
  phonetic?: string;
  literal_translation?: string;
  note?: string;
};

type Desc = {
  id: string;
  source_json: string;
  fps: number;
  theme: 'mint' | 'sunny';
  meta: {
    scene_en: string;
    scene_zh: string;
    task_en: string;
    task_zh: string;
    context: string;
    sentence_zh: string;
    explanation: string;
  };
  cards: Card[];
};

function validateSource(src: any): asserts src is Source {
  const required = [
    'scene_en', 'scene_zh', 'task_en', 'task_zh',
    'context', 'sentence_zh', 'translations', 'explanation',
  ];
  for (const k of required) {
    if (!(k in src)) throw new Error(`源 JSON 缺少字段：${k}`);
  }
  if (!Array.isArray(src.translations) || src.translations.length === 0) {
    throw new Error('源 JSON translations 必须是非空数组');
  }
  for (const t of src.translations) {
    for (const k of ['sentence', 'style', 'note', 'literal_translation', 'phonetic']) {
      if (!(k in t)) throw new Error(`translation 缺少字段：${k}`);
    }
  }
}

function buildCards(src: Source): Card[] {
  const cards: Card[] = [];

  // Card 0: intro（仅中文）
  cards.push({
    index: 0,
    type: 'intro',
    duration_sec: -1,
    tts_segments: [
      { lang: 'zh', text: `${src.scene_zh}：${src.task_zh}` },
    ],
  });

  // Card 1..N: expressions（先中文 literal，再英文 sentence）
  src.translations.forEach((t, i) => {
    cards.push({
      index: i + 1,
      type: 'expression',
      duration_sec: -1,
      tts_segments: [
        { lang: 'zh', text: t.literal_translation },
        { lang: 'en', text: t.sentence },
      ],
      style: t.style,
      sentence_en: t.sentence,
      phonetic: t.phonetic,
      literal_translation: t.literal_translation,
      note: t.note,
    });
  });

  // Card N+1: summary（仅中文）
  cards.push({
    index: cards.length,
    type: 'summary',
    duration_sec: -1,
    tts_segments: [
      { lang: 'zh', text: src.explanation },
    ],
  });

  return cards;
}

function main() {
  const sourcePath = process.argv[2] || DEFAULT_SOURCE;
  console.log(`📂 源 JSON: ${sourcePath}`);

  const raw = JSON.parse(readFileSync(sourcePath, 'utf-8'));
  validateSource(raw);

  const id = basename(sourcePath, '.json');
  const cards = buildCards(raw);

  const desc: Desc = {
    id,
    source_json: sourcePath,
    fps: FPS,
    theme: THEME,
    meta: {
      scene_en: raw.scene_en,
      scene_zh: raw.scene_zh,
      task_en: raw.task_en,
      task_zh: raw.task_zh,
      context: raw.context,
      sentence_zh: raw.sentence_zh,
      explanation: raw.explanation,
    },
    cards,
  };

  mkdirSync(OUTPUT_DIR, { recursive: true });
  const outPath = resolve(OUTPUT_DIR, `${id}.draft.json`);
  writeFileSync(outPath, JSON.stringify(desc, null, 2) + '\n');

  console.log(`✅ 写入 ${outPath}`);
  console.log(`   ${cards.length} 张卡片：${cards.map(c => `${c.index}:${c.type}(${c.tts_segments.length}段)`).join(', ')}`);
  console.log(`   duration_sec 均为 -1，待 generate-card-audio.ts 回填`);
}

main();