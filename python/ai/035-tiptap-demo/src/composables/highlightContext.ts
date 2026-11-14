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
 * 三层分别由 getLayers() 开关控制（右侧工具栏上的三个开关）。
 * 中文框没有词级切分（\S+ 会把一整句当成一个"词"）→ 那一层传 false。
 *
 * 必须是 Decoration 而不是真的 <mark>：编辑器的 HTML 就是翻译接口的入参，
 * 塞真 mark 会污染发给后端的文本和句子对齐。
 */

export const HIGHLIGHT_CONTEXT_KEY = new PluginKey<boolean>('contextHighlight')

/** 三层上下文高亮各自的开关 */
export interface HighlightLayers {
  paragraph: boolean
  sentence: boolean
  word: boolean
}

export interface ContextHighlightOptions {
  /** 用哪套标点切句 */
  splitFn: SplitFn
  /** 对侧光标对齐到的句子索引，-1 = 没有 */
  getCrossIndex: () => number
  /** 三层高亮各自的开关；不传 = 全开 */
  getLayers?: () => HighlightLayers
  /** 正在翻译的句子索引（全局）—— 这些句子画 loading 装饰 */
  getLoadingIndexes?: () => Set<number>
  /** 翻译失败的句子索引（全局）—— 这些句子画 failed 装饰 */
  getFailedIndexes?: () => Set<number>
  /** 鼠标悬停的单词区间 —— 画 hover 装饰 */
  getHoverRange?: () => { from: number; to: number } | null
}

export function createContextHighlightExtension(options: ContextHighlightOptions) {
  const getLayers = options.getLayers ?? (() => ({ paragraph: true, sentence: true, word: true }))

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
              const layers = getLayers()

              if (crossIndex >= 0) {
                // 对侧获焦 → 高亮跨栏对齐到的那一句
                const span = collectSentences(state.doc, options.splitFn).find(
                  (s) => s.index === crossIndex,
                )
                if (span) {
                  if (layers.paragraph) {
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
              } else {
                // 本侧获焦 → 高亮光标自己的段落 / 句子 / 单词
                const ctx = resolveContext(state, options.splitFn)
                if (ctx) {
                  // 空段落不画段落底色 —— 否则编辑器一打开，光标所在的空 <p>
                  // 就顶着一条淡绿底，看起来像凭空多了一行浅色背景
                  if (layers.paragraph && ctx.paragraph.trim()) {
                    decorations.push(
                      Decoration.node(ctx.paraFrom, ctx.paraTo, { class: 'cur-paragraph' }),
                    )
                  }
                  if (layers.sentence && ctx.sentFrom < ctx.sentTo) {
                    decorations.push(
                      Decoration.inline(ctx.sentFrom, ctx.sentTo, { class: 'cur-sentence' }),
                    )
                  }
                  if (layers.word && ctx.wordFrom < ctx.wordTo) {
                    decorations.push(
                      Decoration.inline(ctx.wordFrom, ctx.wordTo, { class: 'cur-word' }),
                    )
                  }
                }
              }

              // ===== 翻译状态 / 悬停 =====
              // 这两种装饰跟「谁获焦」无关，只要集合非空就画。
              const loading = options.getLoadingIndexes?.()
              const failed = options.getFailedIndexes?.()
              const hover = options.getHoverRange?.()

              if ((loading && loading.size) || (failed && failed.size) || hover) {
                const spans = collectSentences(state.doc, options.splitFn)
                for (const span of spans) {
                  if (span.from >= span.to) continue
                  if (loading?.has(span.index)) {
                    decorations.push(
                      Decoration.inline(span.from, span.to, { class: 'sentence-loading' }),
                    )
                  }
                  if (failed?.has(span.index)) {
                    decorations.push(
                      Decoration.inline(span.from, span.to, { class: 'sentence-failed' }),
                    )
                  }
                }
                if (hover && hover.from < hover.to) {
                  decorations.push(
                    Decoration.inline(hover.from, hover.to, { class: 'hover-word' }),
                  )
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
