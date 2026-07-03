// 从 slides.md 抽取每个 v-click 段落的旁白文本。
//
// 设计要点（见记忆 verify-blocks-pattern）：用栈式状态机做 fence 配平，
// 不能复用渲染端正则。fence 围栏内的内容（代码块）整段丢弃，避免把
// `function identity<T>` 之类的代码读进语音里。
//
// 状态：
//   - slideIndex: 当前 slide 序号（1-indexed），`---` 分隔
//   - inFence:   是否在 ``` 围栏内（丢弃）
//   - depth:     v-click <div> 的嵌套层数
//   - buf:       当前 v-click 累积的原始行

import { readFileSync } from 'node:fs';

export type ClickSegment = {
  slide: number;
  click: number;
  text: string;
};

/** 剥掉 markdown 噪音，得到适合 TTS 的纯文本。 */
export function cleanForTTS(raw: string): string {
  let s = raw;
  // 去掉标题井号
  s = s.replace(/^#{1,6}\s+/gm, '');
  // 去掉 HTML 标签（如 <kbd>、<span ...>）
  s = s.replace(/<[^>]+>/g, '');
  // 加粗/斜体 **x** *x* -> x
  s = s.replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1');
  // 行内代码 `x` -> x
  s = s.replace(/`([^`]+)`/g, '$1');
  // 链接 [text](url) -> text
  s = s.replace(/\[([^\]]+)\]\([^)]*\)/g, '$1');
  // 列表符号
  s = s.replace(/^\s*[-*+]\s+/gm, '');
  // 折叠空白，合并成单段
  s = s
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.length > 0)
    .join(' ');
  s = s.replace(/\s{2,}/g, ' ').trim();
  return s;
}

/** 一行是否是围栏起止（``` 或 ~~~，允许前导空白）。 */
function isFence(line: string): boolean {
  return /^\s*(```|~~~)/.test(line);
}

/** 一行是否包含 v-click 起始 div（<div v-click ...>）。 */
function isClickOpen(line: string): boolean {
  return /<div\b[^>]*\bv-click\b[^>]*>/.test(line);
}

/** 一行是否是 </div>（用于配平 v-click 嵌套）。 */
function isDivClose(line: string): boolean {
  return /<\/div>/.test(line);
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
  // 找到闭合的 ---
  let j = i + 1;
  while (j < lines.length && lines[j].trim() !== '---') j += 1;
  // 从闭合行之后重新拼接
  return lines.slice(j + 1).join('\n');
}

/**
 * 解析 markdown 源，抽取所有 v-click 段落文本。
 * 仅处理顶层 `<div v-click>` 包裹的显式块（当前 slides.md 用法）。
 */
export function extractClicks(md: string): ClickSegment[] {
  const lines = stripLeadingFrontmatter(md).split('\n');
  const segments: ClickSegment[] = [];

  let slide = 1;
  let inFence = false;
  let clickDepth = 0; // 当前 v-click 起始 div 之下累积的 <div> 嵌套（含起始那个）
  let buf: string[] = [];
  let clickCounter = 0; // 当前 slide 内 v-click 序号

  const flush = () => {
    const text = cleanForTTS(buf.join('\n'));
    clickCounter += 1;
    if (text) {
      segments.push({ slide, click: clickCounter, text });
    }
    buf = [];
  };

  for (const line of lines) {
    // 围栏切换：即使在 v-click 内部也要吞掉整段代码
    if (isFence(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) {
      // 代码块内容全部丢弃
      continue;
    }

    // slide 分隔（顶层 ---，且不在 v-click 内）
    if (clickDepth === 0 && /^---\s*$/.test(line)) {
      slide += 1;
      clickCounter = 0;
      continue;
    }

    if (clickDepth === 0) {
      // 尚未进入 v-click，寻找起始
      if (isClickOpen(line)) {
        clickDepth = 1;
        buf = [];
      }
      continue;
    }

    // 已在 v-click 内：维护 div 嵌套配平
    if (isClickOpen(line)) {
      clickDepth += 1;
      continue;
    }
    if (isDivClose(line)) {
      clickDepth -= 1;
      if (clickDepth === 0) {
        flush();
      }
      continue;
    }

    buf.push(line);
  }

  return segments;
}

/** 从文件路径读取并解析。 */
export function extractClicksFromFile(path: string): ClickSegment[] {
  return extractClicks(readFileSync(path, 'utf8'));
}
