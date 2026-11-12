import { onBeforeUnmount, ref } from 'vue'
import type { Editor } from '@tiptap/core'

/**
 * 工具栏「复制」按钮的共用逻辑（中英文两个编辑器都要用）。
 *
 * 复制的是编辑器的**纯文本**：段落之间用换行，段内的硬换行（Shift+Enter）
 * 也还原成换行 —— 直接用 getText() 的话硬换行会被吃掉，两行粘成一行。
 */
export function useCopyContent(getEditor: () => Editor | null | undefined) {
  /** 刚复制完（按钮变个色提示一下） */
  const copied = ref(false)
  let timer: ReturnType<typeof setTimeout> | undefined

  async function copyContent() {
    const editor = getEditor()
    if (!editor) return
    const doc = editor.state.doc
    const text = doc.textBetween(
      0,
      doc.content.size,
      '\n',
      (node) => (node.type.name === 'hardBreak' ? '\n' : ''),
    )
    if (!text.trim()) return

    try {
      await navigator.clipboard.writeText(text)
    } catch {
      // 剪贴板 API 不可用（非安全上下文，比如用局域网 IP 访问）→ 退回老办法
      const ta = document.createElement('textarea')
      ta.value = text
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
    }

    copied.value = true
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => {
      copied.value = false
    }, 1500)
  }

  onBeforeUnmount(() => {
    if (timer) clearTimeout(timer)
  })

  return { copied, copyContent }
}
