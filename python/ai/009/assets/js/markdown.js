// markdown.js — marked 配置 + ::: 容器 pre-processor
// 容器类型：quiz, chart, graph, network, train-demo, formula
//
// 实现策略：先扫出所有 ::: type ... ::: 块，替换成占位符 <div data-block-type="..."></div>
// 然后 marked 解析剩余 markdown（占位符是合法 HTML，会被保留）
// 渲染完成后由 ChapterView 扫描这些占位 div，挂载 Vue 组件

const { marked } = window.marked;
const DOMPurify = window.DOMPurify;

export const BLOCK_TYPES = ["quiz", "chart", "graph", "network", "train-demo", "formula"];

// 配置 marked
function configureMarked() {
  marked.setOptions({
    gfm: true,
    breaks: false,
    smartypants: false,
    headerIds: false,
    mangle: false,
  });
}

// ====== 提取 ::: type ... ::: 块 ======
function extractBlocks(mdText) {
  // 匹配 ::: type [args] \n body \n ::: （type 必须是 BLOCK_TYPES 中的一个）
  const typePattern = BLOCK_TYPES.join("|");
  const re = new RegExp(
    `^::: (${typePattern})([^\\n]*)\\n([\\s\\S]*?)\\n:::[ \\t]*$`,
    "gm"
  );
  const blocks = [];   // [{ type, args, body, fullMatch }]
  const replacements = [];  // [original, placeholder, blocks[idx]]

  let m;
  while ((m = re.exec(mdText)) !== null) {
    const type = m[1];
    const args = m[2].trim();
    const body = m[3];
    const idx = blocks.length;
    blocks.push({ type, args, body });
    const placeholder = makePlaceholder(type, args, body, idx);
    replacements.push([m[0], placeholder, idx]);
  }

  // 替换（注意：必须从后往前替换，否则 index 错位）
  let newText = mdText;
  for (let i = replacements.length - 1; i >= 0; i--) {
    const [orig, ph] = replacements[i];
    newText = newText.replace(orig, ph);
  }

  return { text: newText, blocks };
}

function makePlaceholder(type, args, body, idx) {
  // 用 base64 编码 body 防止特殊字符问题
  // 用 data-block-type 标记，ChapterView 扫描后用对应 Vue 组件替换
  const safeArgs = encodeURIComponent(args);
  const safeBody = encodeURIComponent(body);
  // 占位 div 前后各加一个空行，强制 marked 把它当作独立 block。
  // 否则紧跟 `> 引用` 时 marked 的 HTML block 规则会把引用吞进 div 当 inline 文本。
  return `\n\n<div data-block-id="${idx}" data-block-type="${type}" data-block-args="${safeArgs}" data-block-body="${safeBody}"></div>\n\n`;
}

// ====== 解析 quiz 块 ======
export function parseQuizBlock(args, body) {
  const argParts = args.split(/\s+/);
  const id = argParts[0] || `q-${Date.now()}`;
  const type = argParts[1] || "single";
  const placeholderMatch = args.match(/placeholder="([^"]+)"/);
  const placeholder = placeholderMatch ? placeholderMatch[1] : "";

  const lines = body.split("\n");
  const result = { id, type, prompt: "", options: [], answer: "", explain: "", placeholder, modelAnswer: "" };

  let mode = "prompt";
  const promptBuf = [];
  const explainBuf = [];
  const modelAnswerBuf = [];

  for (const line of lines) {
    if (mode === "prompt") {
      if (line.match(/^-\s+[A-Za-z0-9]+:/)) {
        mode = "options";
      } else if (line.match(/^answer\s*:/i)) {
        mode = "answer";
        result.answer = line.replace(/^answer\s*:/i, "").trim();
        continue;
      } else if (line.match(/^model_answer\s*:/i)) {
        mode = "modelAnswer";
        continue;
      } else if (line.match(/^>\s*(.+)$/)) {
        mode = "explain";
        explainBuf.push(line.replace(/^>\s*/, ""));
        continue;
      } else if (line.trim() === "") {
        if (promptBuf.length) promptBuf.push("");
        continue;
      } else {
        promptBuf.push(line);
        continue;
      }
    }

    if (mode === "options") {
      const m = line.match(/^-\s+([A-Za-z0-9]+):\s*(.*)$/);
      if (m) {
        result.options.push({ key: m[1], text: m[2] });
      } else if (line.match(/^answer\s*:/i)) {
        mode = "answer";
        result.answer = line.replace(/^answer\s*:/i, "").trim();
      } else if (line.match(/^model_answer\s*:/i)) {
        mode = "modelAnswer";
      }
    } else if (mode === "answer") {
      if (line.match(/^model_answer\s*:/i)) {
        mode = "modelAnswer";
      } else if (line.match(/^>\s*(.+)$/)) {
        mode = "explain";
        explainBuf.push(line.replace(/^>\s*/, ""));
      } else if (line.trim() === "") {
        // 忽略
      } else {
        result.answer += "," + line.trim();
      }
    } else if (mode === "modelAnswer") {
      if (line.match(/^>\s*(.+)$/)) {
        mode = "explain";
        explainBuf.push(line.replace(/^>\s*/, ""));
      } else if (line.match(/^answer\s*:/i)) {
        // 不太可能
        mode = "answer";
        result.answer = line.replace(/^answer\s*:/i, "").trim();
      } else if (line.trim() === "") {
        if (modelAnswerBuf.length) modelAnswerBuf.push("");
        // 否则忽略开头的空行
      } else {
        modelAnswerBuf.push(line);
      }
    } else if (mode === "explain") {
      if (line.match(/^>\s*(.+)$/)) {
        explainBuf.push(line.replace(/^>\s*/, ""));
      } else if (line.trim() === "") {
        explainBuf.push("");
      }
    }
  }

  result.prompt = promptBuf.join("\n").trim();
  result.explain = explainBuf.join("\n").trim();
  result.modelAnswer = modelAnswerBuf.join("\n").trim();
  result.answer = result.answer.replace(/,+$/, "");
  return result;
}

export function parseChartBlock(args, body) {
  const a = {};
  const re = /(\w+)="([^"]*)"/g;
  let m;
  while ((m = re.exec(args)) !== null) a[m[1]] = m[2];
  return { args: a, body: body.trim() };
}

export function parseGraphBlock(args, body) { return { args, body: body.trim() }; }
export function parseNetworkBlock(args, body) { return { args, body: body.trim() }; }
export function parseTrainDemoBlock(args, body) { return { args, body: body.trim() }; }
export function parseFormulaBlock(args, body) {
  return { body: body.trim(), display: !args.includes("inline") };
}

// ====== 等待 KaTeX 加载 ======
function waitForKatex(maxMs = 3000) {
  return new Promise((resolve) => {
    if (typeof window.katex !== "undefined") return resolve();
    const start = Date.now();
    const timer = setInterval(() => {
      if (typeof window.katex !== "undefined" || Date.now() - start > maxMs) {
        clearInterval(timer);
        resolve();
      }
    }, 30);
  });
}

// ====== 渲染主函数（async 以等待 KaTeX）======
export async function renderMarkdown(mdText) {
  // Step 0: 等 KaTeX 加载
  await waitForKatex();

  // Step 1: 提取 ::: 块，替换成占位符
  const { text: replacedText, blocks } = extractBlocks(mdText);

  // Step 2: 提取 $..$ 和 $$..$$ 公式，KaTeX 预渲染成 HTML 占位符
  const { text: noMath, mathMap } = extractMath(replacedText);

  // Step 3: marked 解析（占位符是合法 HTML，会被保留）
  const rawHtml = marked.parse(noMath);

  // Step 4: 把 mathMap 里的占位符替换回 KaTeX 渲染的 HTML
  const finalHtml = Object.keys(mathMap).reduce((html, key) => {
    return html.split(key).join(mathMap[key]);
  }, rawHtml);

  // Step 5: XSS 清洗
  const clean = DOMPurify.sanitize(finalHtml, {
    ADD_TAGS: ["div", "span", "math", "annotation", "semantics", "mrow", "mi", "mo", "mn", "msup", "msub", "mfrac", "msqrt", "mtable", "mtr", "mtd", "mover", "munder", "munderover", "mspace", "mtext", "mstyle", "mpadded", "menclose", "mstyle", "svg", "path", "g", "rect", "line"],
    ADD_ATTR: ["data-block-type", "data-block-args", "data-block-body", "data-block-id", "class", "style", "aria-hidden", "role", "xmlns"],
  });

  return { html: clean, blocks };
}

// 提取数学公式并用 KaTeX 预渲染
function extractMath(mdText) {
  const mathMap = {};
  let counter = 0;

  // 先匹配块级 $$..$$（贪婪匹配）
  mdText = mdText.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
    const key = `@@MATH_BLOCK_${counter++}@@`;
    mathMap[key] = renderKatex(tex, true);
    return key;
  });

  // 再匹配行内 $..$
  mdText = mdText.replace(/\$([^\$\n]+?)\$/g, (_, tex) => {
    const key = `@@MATH_INLINE_${counter++}@@`;
    mathMap[key] = renderKatex(tex, false);
    return key;
  });

  return { text: mdText, mathMap };
}

function renderKatex(tex, display) {
  if (typeof window.katex === "undefined") {
    return display ? `$$${tex}$$` : `$${tex}$`;
  }
  try {
    return window.katex.renderToString(tex, { throwOnError: false, displayMode: display });
  } catch (e) {
    return display ? `$$${tex}$$` : `$${tex}$`;
  }
}

// 启动
configureMarked();
