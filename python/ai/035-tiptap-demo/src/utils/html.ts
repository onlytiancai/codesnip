/**
 * 一点点 HTML 工具 —— 只服务于「粘贴去格式」和「译文写回编辑器」。
 */

/** HTML 转义 —— 拼进 innerHTML 的文本不能带着 < > & 进去。 */
export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/**
 * 会另起一段的标签 —— 遇到它们就断一行。
 *
 * 粘贴进来的网页/Word 内容里，段落结构全靠这些标签；
 * 我们只要「文字 + 段落划分」，其余（加粗、代码块、引用、列表…）全丢。
 */
const BLOCK_TAGS = new Set([
  'P', 'DIV', 'LI', 'UL', 'OL', 'DL', 'DT', 'DD',
  'TR', 'TD', 'TH', 'TABLE', 'THEAD', 'TBODY', 'TFOOT',
  'H1', 'H2', 'H3', 'H4', 'H5', 'H6',
  'BLOCKQUOTE', 'PRE', 'SECTION', 'ARTICLE', 'ASIDE', 'NAV',
  'HEADER', 'FOOTER', 'MAIN', 'FIGURE', 'FIGCAPTION', 'HR', 'FORM',
])

/**
 * 粘贴的 HTML → 纯文本段落 HTML。
 *
 * 加粗 / 斜体 / 下划线 / 代码块 / 引用 / 列表 / 表格全部丢掉，
 * 只保留文字和「哪几行是同一段」。
 *
 * 为什么不直接读剪贴板里的 text/plain：
 * transformPastedHTML 拿到的是**已经渲染过的** HTML，
 * 在这里统一洗一遍最省事 —— 不用改 ProseMirror 的粘贴流程。
 */
export function htmlToPlainParagraphs(html: string): string {
  if (!html) return ''

  const doc = new DOMParser().parseFromString(html, 'text/html')
  const lines: string[] = []
  let buf = ''

  const walk = (node: Node) => {
    node.childNodes.forEach((child) => {
      // 文本
      if (child.nodeType === Node.TEXT_NODE) {
        // 不换行空格当普通空格 —— 复制网页时到处都是
        buf += (child.textContent ?? '').replace(/\u00a0/g, ' ')
        return
      }
      if (child.nodeType !== Node.ELEMENT_NODE) return

      const el = child as HTMLElement
      const tag = el.tagName
      // 脚本 / 样式的内容不是正文
      if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT') return

      // 软换行 → 断行
      if (tag === 'BR') {
        lines.push(buf)
        buf = ''
        return
      }

      const isBlock = BLOCK_TAGS.has(tag)
      if (isBlock) {
        lines.push(buf)
        buf = ''
      }
      walk(el)
      if (isBlock) {
        lines.push(buf)
        buf = ''
      }
    })
  }

  walk(doc.body)
  lines.push(buf)

  // 空行丢掉（粘贴进来的网页经常一堆 <div><br></div>）
  const paras = lines.map((l) => l.trim()).filter((l) => l.length > 0)
  return paras.map((p) => `<p>${escapeHtml(p)}</p>`).join('')
}
