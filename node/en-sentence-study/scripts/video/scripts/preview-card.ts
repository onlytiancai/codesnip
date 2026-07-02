// 把 desc JSON 中的指定卡片渲染成 HTML 预览
// 用法：
//   pnpm exec tsx scripts/preview-card.ts                       # 默认 desc/1.draft.json 第 1 张
//   pnpm exec tsx scripts/preview-card.ts path/to/desc.json 2  # 指定 desc + card index
//
// 输出：scripts/preview/<desc-id>-card-<idx>.html，用浏览器打开预览

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

const DEFAULT_DESC = resolve(ROOT, 'scripts/desc/1.draft.json');
const OUTPUT_DIR = resolve(ROOT, 'scripts/preview');

const STYLE_COLORS: Record<string, { bg: string; text: string; label: string }> = {
  polite:  { bg: '#A8E6CF', text: '#0E3B2E', label: 'POLITE · 礼貌' },
  neutral: { bg: '#CFE4F5', text: '#0E3B2E', label: 'NEUTRAL · 中性' },
  casual:  { bg: '#FFD6A8', text: '#3E2A14', label: 'CASUAL · 口语' },
  bold:    { bg: '#F5B7C5', text: '#3E2A14', label: 'BOLD · 直接' },
};

function esc(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderExpressionBody(card: any): string {
  const sc = STYLE_COLORS[card.style || 'neutral'];
  return `
    <div class="style-badge" style="background:${sc.bg}; color:${sc.text}">${esc(sc.label)}</div>
    <div class="zh-line">${esc(card.literal_translation || '')}</div>
    <div class="en-line">${esc(card.sentence_en || '')}</div>
    <div class="phonetic">/${esc(card.phonetic || '')}/</div>
    <div class="note">${esc(card.note || '')}</div>
  `;
}

function renderIntroBody(card: any, meta: any): string {
  return `
    <div class="intro-eyebrow">SCENE</div>
    <h1 class="intro-title">${esc(meta.scene_zh)}</h1>
    <div class="intro-divider"></div>
    <div class="intro-eyebrow">TASK</div>
    <h2 class="intro-task">${esc(meta.task_zh)}</h2>
    <div class="intro-sentence">${esc(meta.sentence_zh)}</div>
  `;
}

function renderSummaryBody(card: any, meta: any): string {
  return `
    <div class="summary-eyebrow">SUMMARY · 今日小结</div>
    <p class="summary-text">${esc(meta.explanation || '')}</p>
  `;
}

function renderBody(card: any, meta: any): string {
  if (card.type === 'expression') return renderExpressionBody(card);
  if (card.type === 'intro') return renderIntroBody(card, meta);
  if (card.type === 'summary') return renderSummaryBody(card, meta);
  return '';
}

function renderHtml(desc: any, card: any): string {
  const durLabel = card.duration_sec === -1
    ? '⏳ 音频未生成（duration_sec = -1）'
    : `⏱ ${card.duration_sec}s · ${card.tts_segments.length} 段 TTS`;
  const segs = card.tts_segments
    .map((s: any, i: number) => `<li><b>${s.lang}</b>: ${esc(s.text)}</li>`)
    .join('');
  const bg = '#F4FBF8';

  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<title>Card ${card.index} Preview · ${esc(desc.meta.scene_zh)} · ${esc(desc.meta.task_zh)}</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: #2A2A2A;
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Helvetica Neue", "Microsoft YaHei", sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    padding: 24px;
  }
  .meta {
    position: fixed;
    top: 12px;
    left: 12px;
    right: 12px;
    color: #DDD;
    font-size: 13px;
    line-height: 1.5;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    pointer-events: none;
    z-index: 100;
  }
  .meta b { color: #19A974; }
  .meta ul { margin: 4px 0 0 16px; padding: 0; }
  .phone {
    width: 375px;
    height: 812px;
    background: ${bg};
    border-radius: 44px;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    position: relative;
  }
  /* 内部 stage 用 1080x1920 原生尺寸，缩放 0.347 适配 375x667 */
  .stage {
    position: absolute;
    width: 1080px;
    height: 1920px;
    transform-origin: top left;
    transform: scale(0.347);
  }

  /* ── Header / Footer ── */
  .header, .footer {
    position: absolute;
    left: 0; right: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(8px);
  }
  .header {
    top: 0; height: 110px;
    border-bottom: 2px solid #CFE9DC;
  }
  .footer {
    bottom: 0; height: 90px;
    border-top: 2px solid #CFE9DC;
    font-size: 26px; color: #5A8475;
    letter-spacing: 1px;
  }
  .header-content { display: flex; align-items: center; }
  .header-bar {
    width: 12px; height: 36px;
    background: #19A974;
    border-radius: 6px;
    margin-right: 16px;
  }
  .header-text {
    font-size: 38px; color: #0E3B2E;
    font-weight: 700; letter-spacing: 2px;
  }

  /* ── 卡片区 ── */
  .card-area {
    position: absolute;
    top: 110px; bottom: 90px;
    left: 0; right: 0;
    padding: 100px 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }

  /* ── Expression 卡片 ── */
  .style-badge {
    display: inline-block;
    padding: 16px 36px;
    border-radius: 100px;
    font-size: 36px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 80px;
    width: fit-content;
  }
  .zh-line {
    font-size: 92px;
    font-weight: 600;
    color: #0E3B2E;
    line-height: 1.25;
    margin-bottom: 50px;
    word-wrap: break-word;
    word-break: break-word;
  }
  .en-line {
    font-size: 76px;
    font-weight: 600;
    color: #19A974;
    line-height: 1.25;
    margin-bottom: 40px;
    word-wrap: break-word;
    word-break: break-word;
  }
  .phonetic {
    font-size: 44px;
    color: #5A8475;
    font-style: italic;
    margin-bottom: 60px;
    word-wrap: break-word;
  }
  .note {
    font-size: 42px;
    color: #5A8475;
    line-height: 1.5;
    padding: 30px 40px;
    background: rgba(207, 233, 220, 0.4);
    border-radius: 24px;
    border-left: 8px solid #19A974;
    word-wrap: break-word;
  }

  /* ── Intro 卡片 ── */
  .intro-eyebrow {
    font-size: 36px;
    color: #19A974;
    font-weight: 600;
    letter-spacing: 6px;
    margin-bottom: 24px;
  }
  .intro-title {
    font-size: 160px;
    color: #0E3B2E;
    font-weight: 800;
    margin: 0 0 24px 0;
    line-height: 1.1;
    word-wrap: break-word;
  }
  .intro-divider {
    width: 120px;
    height: 6px;
    background: #19A974;
    margin: 60px 0;
    border-radius: 3px;
  }
  .intro-task {
    font-size: 100px;
    color: #19A974;
    font-weight: 700;
    margin: 0 0 80px 0;
    line-height: 1.2;
    word-wrap: break-word;
  }
  .intro-sentence {
    font-size: 56px;
    color: #5A8475;
    line-height: 1.4;
    word-wrap: break-word;
  }

  /* ── Summary 卡片 ── */
  .summary-eyebrow {
    font-size: 44px;
    color: #19A974;
    font-weight: 700;
    letter-spacing: 4px;
    margin-bottom: 60px;
  }
  .summary-text {
    font-size: 64px;
    color: #0E3B2E;
    line-height: 1.6;
    word-wrap: break-word;
    word-break: break-word;
    margin: 0;
  }
</style>
</head>
<body>
<div class="meta">
  <b>Card ${card.index}</b> · type=${card.type} · ${durLabel}<br/>
  <b>Desc</b>: ${esc(desc.id)} · theme=${esc(desc.theme)}<br/>
  <b>TTS segments</b>: <ul>${segs}</ul>
</div>

<div class="phone">
  <div class="stage">
    <div class="header">
      <div class="header-content">
        <div class="header-bar"></div>
        <span class="header-text">英语口语 · 每日一句</span>
      </div>
    </div>

    <div class="card-area">
      ${renderBody(card, desc.meta)}
    </div>

    <div class="footer">@en-sentence-study</div>
  </div>
</div>

</body>
</html>
`;
}

function main() {
  const descPath = process.argv[2] || DEFAULT_DESC;
  const cardIdx = parseInt(process.argv[3] || '1', 10);

  const desc = JSON.parse(readFileSync(descPath, 'utf-8'));
  const card = desc.cards.find((c: any) => c.index === cardIdx);
  if (!card) {
    console.error(`❌ desc 中没有 card index=${cardIdx}`);
    process.exit(1);
  }

  mkdirSync(OUTPUT_DIR, { recursive: true });
  const outPath = resolve(OUTPUT_DIR, `${desc.id}-card-${cardIdx}.html`);
  writeFileSync(outPath, renderHtml(desc, card));
  console.log(`✅ 写入 ${outPath}`);
  console.log(`   在浏览器打开：file://${outPath}`);
}

main();