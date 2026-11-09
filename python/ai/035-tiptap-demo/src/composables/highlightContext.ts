import { Extension } from '@tiptap/core'
import { Plugin, PluginKey } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'
import type { SplitFn } from './useWritingContext'

/**
 * 跨面板句子联动高亮：
 * - 给某个 Tiptap editor 注册一个 Extension
 * - 它在 ProseMirror 里渲染一个 Decoration，包裹第 highlightedSentenceIndex 句
 * - getIndex 是个 closure，外层 ref 变化后下一次 transaction 会重新算 decorations
 */
export const HIGHLIGHT_SENTENCE_KEY = new PluginKey<boolean>('highlightSentence')

export function createHighlightSentenceExtension(splitFn: SplitFn, getIndex: () => number) {
  return Extension.create({
    name: 'highlightSentence',

    addProseMirrorPlugins() {
      return [
        new Plugin({
          key: HIGHLIGHT_SENTENCE_KEY,

          props: {
            decorations(state) {
              const idx = getIndex()
              if (idx < 0) return null

              const decorations: Decoration[] = []
              let globalIdx = 0

              state.doc.descendants((node, pos) => {
                if (node.type.name !== 'paragraph' && node.type.name !== 'heading') return
                const sentences = splitFn(node.textContent)
                for (const s of sentences) {
                  if (globalIdx === idx) {
                    const from = pos + 1 + s.start // +1 跳过 paragraph opening
                    const end = pos + 1 + s.end
                    if (from < end) {
                      decorations.push(
                        Decoration.inline(from, end, {
                          class: 'cross-sentence-highlight',
                        }),
                      )
                    }
                  }
                  globalIdx++
                }
              })

              return decorations.length > 0
                ? DecorationSet.create(state.doc, decorations)
                : null
            },
          },
        }),
      ]
    },
  })
}

/**
 * 强制让 ProseMirror 重新评估 decorations。
 * 在 highlightIndex ref 变化后调一次即可，下一次 transaction 会自然跑。
 */
export function rerenderDecorations(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  editor: any,
) {
  editor.view.dispatch(editor.state.tr.setMeta(HIGHLIGHT_SENTENCE_KEY, true))
}
