import { Extension } from '@tiptap/core'
import { Plugin, PluginKey } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'
import {
  collectSentences,
  resolveContext,
  type SplitFn,
} from './useWritingContext'

/**
 * 编辑器内的高亮装饰。
 *
 * 一个编辑器里同时最多画一套高亮，靠「对侧索引」决定画哪一套：
 *
 * - 对侧获焦（getCrossIndex() >= 0）：画**跨栏对齐**的那一句（紫色）+ 它所在段落（淡绿）
 * - 本侧获焦（getCrossIndex() < 0）：画光标自己的段落（淡绿）/ 句子（淡蓝）/ 单词（黄）
 *
 * 中文框只要紫色的跨栏高亮 → showContextLayers 传 false；
 * 英文框三层都要 → 传 true。
 *
 * 必须是 Decoration 而不是真的 <mark>：编辑器的 HTML 就是翻译接口的入参，
 * 塞真 mark 会污染发给后端的文本和句子对齐。
 */

export const HIGHLIGHT_CONTEXT_KEY = new PluginKey<boolean>('contextHighlight')

export interface ContextHighlightOptions {
  /** 用哪套标点切句 */
  splitFn: SplitFn
  /** 对侧光标对齐到的句子索引，-1 = 没有 */
  getCrossIndex: () => number
  /** 是否画「上下文三层」（段落 / 句子 / 单词）。中文框传 false */
  showContextLayers?: () => boolean
}

export function createContextHighlightExtension(options: ContextHighlightOptions) {
  const showLayers = options.showContextLayers ?? (() => true)

  return Extension.create({
    name: 'contextHighlight',

    addProseMirrorPlugins() {
      return [
        new Plugin({
          key: HIGHLIGHT_CONTEXT_KEY,

          props: {
            decorations(state) {
              const decorations: Decoration[] = []
              const crossIndex = options.getCrossIndex()

              if (crossIndex >= 0) {
                // 对侧获焦 → 高亮跨栏对齐到的那一句
                const span = collectSentences(state.doc, options.splitFn).find(
                  (s) => s.index === crossIndex,
                )
                if (span) {
                  if (showLayers()) {
                    decorations.push(
                      Decoration.node(span.paraFrom, span.paraTo, {
                        class: 'cur-paragraph',
                      }),
                    )
                  }
                  if (span.from < span.to) {
                    decorations.push(
                      Decoration.inline(span.from, span.to, {
                        class: 'cross-sentence-highlight',
                      }),
                    )
                  }
                }
              } else if (showLayers()) {
                // 本侧获焦 → 高亮光标自己的段落 / 句子 / 单词
                const ctx = resolveContext(state, options.splitFn)
                if (ctx) {
                  // 空段落不画段落底色 —— 否则编辑器一打开，光标所在的空 <p>
                  // 就顶着一条淡绿底，看起来像凭空多了一行浅色背景
                  if (ctx.paragraph.trim()) {
                    decorations.push(
                      Decoration.node(ctx.paraFrom, ctx.paraTo, { class: 'cur-paragraph' }),
                    )
                  }
                  if (ctx.sentFrom < ctx.sentTo) {
                    decorations.push(
                      Decoration.inline(ctx.sentFrom, ctx.sentTo, { class: 'cur-sentence' }),
                    )
                  }
                  if (ctx.wordFrom < ctx.wordTo) {
                    decorations.push(
                      Decoration.inline(ctx.wordFrom, ctx.wordTo, { class: 'cur-word' }),
                    )
                  }
                }
              }

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
 * 在 getCrossIndex 变化后调一次即可（跨栏索引来自另一个编辑器，
 * 改 prop 不会自动触发 transaction）。
 */
export function rerenderDecorations(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  editor: any,
) {
  editor.view.dispatch(editor.state.tr.setMeta(HIGHLIGHT_CONTEXT_KEY, true))
}
