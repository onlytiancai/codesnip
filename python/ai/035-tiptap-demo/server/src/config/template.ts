/**
 * 极简 mustache 渲染器（约 130 行，不引第三方依赖）。
 *
 * 支持：
 *   {{name}}              变量替换，缺失 → 空字符串
 *   {{#name}}…{{/name}}   变量为真值时渲染整块（非空字符串 / 非 false / 非 0）
 *   {{^name}}…{{/name}}   变量为假值时渲染整块
 *   {{! … }}              注释，忽略
 *   {{a.b}}               点号路径
 *
 * **standalone 行处理**（mustache 标准语义）：
 * section 标签独占一行时，只把这一行里除标签以外的内容去掉——
 * 标签前的缩进 + 标签后的换行。所以
 *
 *     Rules:
 *     {{#kind}}
 *     - one
 *     {{/kind}}
 *     - two
 *
 * 渲染成 `Rules:\n- one\n- two`（kind 为真时），不会多出空行。
 *
 * 这条必须做对：提示词是从旧代码逐字迁移过来的，渲染结果要和老 builder
 * 逐字节相同，否则迁移就不是「零行为变化」了。
 */

export type TemplateVars = Record<string, unknown>

type Token =
  | { kind: 'text'; value: string }
  | { kind: 'var'; name: string }
  | { kind: 'open'; name: string; inverted: boolean }
  | { kind: 'close'; name: string }

type Node =
  | { kind: 'text'; value: string }
  | { kind: 'var'; name: string }
  | { kind: 'section'; name: string; inverted: boolean; children: Node[] }

const TAG_RE =
  /\{\{![\s\S]*?\}\}|\{\{([#^])\s*([\w.]+)\s*\}\}|\{\{\/\s*([\w.]+)\s*\}\}|\{\{\{?\s*([\w.]+)\s*\}?\}\}/g

function lookup(vars: TemplateVars, path: string): unknown {
  if (!path.includes('.')) return vars[path]
  let cur: unknown = vars
  for (const part of path.split('.')) {
    if (cur === null || typeof cur !== 'object') return undefined
    cur = (cur as Record<string, unknown>)[part]
  }
  return cur
}

/** 假值判定：空字符串 / false / 0 / null / undefined 都算假 */
function isFalsy(value: unknown): boolean {
  if (value === undefined || value === null || value === false) return true
  if (typeof value === 'string') return value.length === 0
  if (typeof value === 'number') return value === 0
  return false
}

function stringify(value: unknown): string {
  if (value === undefined || value === null) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value)
}

function tokenize(src: string): Token[] {
  const tokens: Token[] = []
  let last = 0
  TAG_RE.lastIndex = 0
  let m: RegExpExecArray | null
  while ((m = TAG_RE.exec(src)) !== null) {
    if (m.index > last) tokens.push({ kind: 'text', value: src.slice(last, m.index) })
    const [full, sigil, name, closeName, varName] = m
    if (full.startsWith('{{!')) {
      // 注释：直接丢掉
    } else if (sigil) {
      tokens.push({ kind: 'open', name: name as string, inverted: sigil === '^' })
    } else if (closeName) {
      tokens.push({ kind: 'close', name: closeName })
    } else if (varName) {
      tokens.push({ kind: 'var', name: varName })
    }
    last = m.index + full.length
  }
  if (last < src.length) tokens.push({ kind: 'text', value: src.slice(last) })
  return tokens
}

/** 某段文本里最后一个 \n 之后的部分（标签所在行的前半截） */
function tailLine(text: string): string {
  return text.slice(text.lastIndexOf('\n') + 1)
}

/** 某段文本里第一个 \n 之前的部分（标签所在行的后半截） */
function headLine(text: string): string {
  const i = text.indexOf('\n')
  return i === -1 ? text : text.slice(0, i)
}

/**
 * 标签是否独占一行：
 * - 同一行在标签之前只有空白
 * - 同一行在标签之后只有空白
 */
function isStandalone(prev: Token | undefined, next: Token | undefined): boolean {
  if (prev === undefined) {
    // 模板开头，行首就是标签
  } else if (prev.kind === 'text') {
    if (tailLine(prev.value).trim() !== '') return false
  } else {
    return false // 同一行前面还有别的标签 → 不算 standalone
  }

  if (next === undefined) {
    // 模板结尾
  } else if (next.kind === 'text') {
    if (headLine(next.value).trim() !== '') return false
  } else {
    return false
  }

  return true
}

/** 按 standalone 规则调整相邻文本 token：抹掉标签所在行的空白部分。 */
function applyStandalone(tokens: Token[]): void {
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i]
    if (t.kind !== 'open' && t.kind !== 'close') continue
    if (!isStandalone(tokens[i - 1], tokens[i + 1])) continue

    const prev = tokens[i - 1]
    if (prev && prev.kind === 'text') {
      // 去掉标签前的行内缩进（不动前一个换行）
      prev.value = prev.value.replace(/[ \t]+$/, '')
    }

    const next = tokens[i + 1]
    if (next && next.kind === 'text') {
      if (next.value.includes('\n')) {
        // 去掉标签后那一整行（含换行）
        next.value = next.value.replace(/^[ \t]*\r?\n/, '')
      } else if (next.value.trim() === '') {
        // 标签后只剩空白且没有换行 → 整段删掉
        next.value = ''
      }
    }
  }
}

function buildTree(tokens: Token[]): Node[] {
  const root: Node[] = []
  const stack: Node[][] = [root]
  for (const t of tokens) {
    const current = stack[stack.length - 1]
    if (t.kind === 'open') {
      const node: Node = {
        kind: 'section',
        name: t.name,
        inverted: t.inverted,
        children: [],
      }
      current.push(node)
      stack.push(node.children)
    } else if (t.kind === 'close') {
      if (stack.length > 1) stack.pop()
    } else {
      current.push(t)
    }
  }
  return root
}

function renderNodes(nodes: Node[], vars: TemplateVars): string {
  let out = ''
  for (const n of nodes) {
    if (n.kind === 'text') {
      out += n.value
    } else if (n.kind === 'var') {
      out += stringify(lookup(vars, n.name))
    } else {
      const value = lookup(vars, n.name)
      const falsy = isFalsy(value)
      const keep = n.inverted ? falsy : !falsy
      if (keep) out += renderNodes(n.children, vars)
    }
  }
  return out
}

/**
 * 渲染模板。
 *
 * @param template 模板文本
 * @param vars     变量表
 */
export function render(template: string, vars: TemplateVars): string {
  const tokens = tokenize(template)
  applyStandalone(tokens)
  return renderNodes(buildTree(tokens), vars)
}

/**
 * 模板是否括号配对。用来在保存提示词时挡住写坏的模板。
 */
export function isBalanced(template: string): boolean {
  const opens = (template.match(/\{\{[#^]\s*[\w.]+\s*\}\}/g) ?? []).length
  const closes = (template.match(/\{\{\/\s*[\w.]+\s*\}\}/g) ?? []).length
  if (opens !== closes) return false

  // 去掉所有合法标签后不该再剩下 {{ 或 }}
  const stripped = template
    .replace(/\{\{![\s\S]*?\}\}/g, '')
    .replace(/\{\{[#^]\s*[\w.]+\s*\}\}/g, '')
    .replace(/\{\{\/\s*[\w.]+\s*\}\}/g, '')
    .replace(/\{\{\{?\s*[\w.]+\s*\}?\}\}/g, '')
  return !stripped.includes('{{') && !stripped.includes('}}')
}
