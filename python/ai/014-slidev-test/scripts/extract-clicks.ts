// 从 slides.md 抽取每个 v-click 的旁白文本。
//
// 旁白用「显式语法」定义：在 slide 里写 HTML 注释指令。
//
//   <!-- narrate: 文本 -->        自动编号（同一 slide 内按出现顺序 1,2,3...）
//   <!-- narrate 3: 文本 -->      显式指定 click 序号（推荐用于复杂 slide）
//
// 为什么支持显式序号：当一张 slide 的 click 来源混杂（v-click 元素、
// clicks: 帧数、$clicks 高亮步骤等），靠「数 div」无法对齐 Slidev 实际的
// click index。显式写 `narrate N:` 让旁白精确绑定到第 N 个 click。
//
// 规则：
//   - slide 序号：headmatter 之后，按顶层 `---` 分隔递增
//   - 代码围栏（``` / ~~~）内整段忽略，示例代码里的注释不会被误当旁白
//   - narrate 指令可跨行；文本内多余空白会折叠成单空格

import { readFileSync } from 'node:fs';

export type ClickSegment = {
  slide: number;
  click: number;
  text: string;
};

/** 全局匹配 narrate 指令，捕获可选序号(1)与文本(2)。 */
const NARRATE_RE = /<!--\s*narrate\s*(\d+)?\s*:\s*([\s\S]*?)-->/gi;

/** 一行是否是围栏起止（``` 或 ~~~，允许前导空白）。 */
function isFence(line: string): boolean {
  return /^\s*(```|~~~)/.test(line);
}

/**
 * 剥掉文件开头的 headmatter（deck 配置）。
 * headmatter 是文件最顶端 `---` ... `---` 包裹的块，不算 slide 分隔。
 */
function stripLeadingFrontmatter(md: string): string {
  const lines = md.split('\n');
  let i = 0;
  while (i < lines.length && lines[i].trim() === '') i += 1;
  if (i >= lines.length || lines[i].trim() !== '---') {
    return md; // 没有 headmatter
  }
  let j = i + 1;
  while (j < lines.length && lines[j].trim() !== '---') j += 1;
  return lines.slice(j + 1).join('\n');
}

/** 从一张 slide 的正文（已去掉围栏代码）里抽出全部 narrate 段。 */
function extractFromSlide(slide: number, body: string): ClickSegment[] {
  const out: ClickSegment[] = [];
  let auto = 0;
  NARRATE_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = NARRATE_RE.exec(body)) !== null) {
    const text = m[2].replace(/\s+/g, ' ').trim();
    if (!text) continue;
    let click: number;
    if (m[1] != null) {
      click = Number(m[1]);
    } else {
      auto += 1;
      click = auto;
    }
    out.push({ slide, click, text });
  }
  return out;
}

/**
 * 解析 markdown 源，抽取所有 narrate 旁白段。
 */
export function extractClicks(md: string): ClickSegment[] {
  const lines = stripLeadingFrontmatter(md).split('\n');
  const segments: ClickSegment[] = [];

  let slide = 1;
  let inFence = false;
  let buf: string[] = []; // 当前 slide 的正文（不含围栏代码）

  const flush = () => {
    segments.push(...extractFromSlide(slide, buf.join('\n')));
    buf = [];
  };

  for (const line of lines) {
    if (isFence(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) {
      continue; // 代码块内容整段丢弃
    }
    if (/^---\s*$/.test(line)) {
      flush();
      slide += 1;
      continue;
    }
    buf.push(line);
  }
  flush();

  // 按 slide、click 排序，保证输出稳定
  segments.sort((a, b) => a.slide - b.slide || a.click - b.click);
  return segments;
}

/** 从文件路径读取并解析。 */
export function extractClicksFromFile(path: string): ClickSegment[] {
  return extractClicks(readFileSync(path, 'utf8'));
}
