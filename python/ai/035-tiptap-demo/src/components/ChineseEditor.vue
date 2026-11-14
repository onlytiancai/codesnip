<script setup lang="ts">
import { ref, watch } from 'vue'
import { EditorContent, useEditor } from '@tiptap/vue-3'
import StarterKit from '@tiptap/starter-kit'
import {
  getSentenceIndex,
  splitChineseSentences,
  topBlockContentRanges,
} from '../composables/useWritingContext'
import {
  createContextHighlightExtension,
  rerenderDecorations,
} from '../composables/highlightContext'
import { useCopyContent } from '../composables/useCopyContent'
import { escapeHtml, htmlToPlainParagraphs } from '../utils/html'
import EditorIcon from './EditorIcon.vue'

const props = withDefaults(
  defineProps<{
    /** 来自英文侧的高亮：英文句子 i 时，此处把中文第 i 句高亮 */
    highlightedZhIndex: number
    /**
     * 是不是源侧。源侧给「清空」，译侧给「复制」——
     * 切方向时两个框的按钮跟着互换。
     */
    isSource?: boolean
    /** 正在翻译的句子（全局句子索引）—— 这些句子显示 loading */
    loadingIndexes?: number[]
    /** 翻译失败的句子（全局句子索引） */
    failedIndexes?: number[]
    /** 能不能编辑。翻译期间会被锁住 */
    editable?: boolean
  }>(),
  { isSource: true, loadingIndexes: () => [], failedIndexes: () => [], editable: true },
)

const html = defineModel<string>('html', { default: '' })
/** 光标所在句子的文档级全局索引（跨面板对齐的唯一依据） */
const sentenceIndex = defineModel<number>('sentenceIndex', { default: -1 })

// 闭包给 ProseMirror 读
const highlightRef = ref(props.highlightedZhIndex)
watch(
  () => props.highlightedZhIndex,
  (v) => {
    highlightRef.value = v
    if (editor.value) rerenderDecorations(editor.value)
  },
)

// ===== 翻译中的句子 =====
/** 正在翻的句子（全局索引）—— 给装饰扩展读 */
let loadingSet = new Set<number>()
/** 翻失败的句子（全局索引） */
let failedSet = new Set<number>()

watch(
  () => props.loadingIndexes,
  (v) => {
    loadingSet = new Set(v)
    if (editor.value) rerenderDecorations(editor.value)
  },
)
watch(
  () => props.failedIndexes,
  (v) => {
    failedSet = new Set(v)
    if (editor.value) rerenderDecorations(editor.value)
  },
)

// ===== 翻译期间锁住编辑器 =====
watch(
  () => props.editable,
  (v) => {
    editor.value?.setEditable(v)
  },
)

// 父组件把译文整体写进来（英译中模式下中文栏是译侧）
defineExpose({
  setHtml(html: string) {
    editor.value?.commands.setContent(html)
  },

  /**
   * 把译文按段落写回（逐句显示用）。
   *
   * 只替换**变了的那几段**，而且从后往前改 —— 改后面的段落不会影响前面段落的
   * doc 位置。整篇 setContent 的话译文框会闪一下、滚动位置还会跳回顶部。
   * 段落数对不上（用户手动加过段落）就整体重建。
   */
  applyParagraphs(paragraphs: string[], changed: number[]) {
    const ed = editor.value
    if (!ed) return
    const ranges = topBlockContentRanges(ed.state.doc)
    if (ranges.length !== paragraphs.length) {
      ed.commands.setContent(paragraphs.map((t) => `<p>${escapeHtml(t)}</p>`).join(''))
      return
    }
    const tr = ed.state.tr
    for (const p of [...changed].sort((a, b) => b - a)) {
      const r = ranges[p]
      if (!r) continue
      tr.insertText(paragraphs[p], r.from, r.to)
    }
    ed.view.dispatch(tr)
  },
})

/** 编辑区有没有焦点 —— 撤销/重做只在聚焦时显示 */
const focused = ref(false)

const editor = useEditor({
  content: '<p></p>',
  extensions: [
    StarterKit.configure({
      heading: false,
      codeBlock: false,
      horizontalRule: false,
      blockquote: false,
      bulletList: false,
      orderedList: false,
    }),
    createContextHighlightExtension({
      splitFn: splitChineseSentences,
      getCrossIndex: () => highlightRef.value,
      // 中文框只要紫色的跨栏高亮，不要段落/句子/单词三层
      showContextLayers: () => false,
      getLoadingIndexes: () => loadingSet,
      getFailedIndexes: () => failedSet,
    }),
  ],
  editorProps: {
    /**
     * 粘贴 → 纯文本。
     *
     * 从 Word / 网页复制过来的一堆加粗、代码块、引用、列表全丢掉，
     * 只留文字和段落划分 —— 中文框只是译文输入，不需要格式。
     */
    transformPastedHTML: (html: string) => htmlToPlainParagraphs(html),
  },
  onUpdate({ editor }) {
    html.value = editor.getHTML()
    sentenceIndex.value = getSentenceIndex(editor.state, splitChineseSentences)
  },
  onSelectionUpdate({ editor }) {
    sentenceIndex.value = getSentenceIndex(editor.state, splitChineseSentences)
  },
  // 聚焦状态 —— 撤销/重做只在输入框被点着的时候才露出来
  onFocus() {
    focused.value = true
  },
  onBlur() {
    focused.value = false
  },
  onCreate({ editor }) {
    editor.setEditable(props.editable)
    html.value = editor.getHTML()
    sentenceIndex.value = getSentenceIndex(editor.state, splitChineseSentences)
  },
})

// ===== 复制内容 =====
const { copied, copyContent } = useCopyContent(() => editor.value)
</script>

<template>
  <div class="editor-shell zh-shell" :class="{ 'is-locked': !editable }">
    <div v-if="editor" class="toolbar">
      <!-- 左：撤销 / 重做 —— 只在输入框被点着（聚焦）的时候才显示。
           按钮上 @mousedown.prevent —— 不然鼠标一点按钮，焦点就跑了，
           按钮当场消失，点一下像没点着。 -->
      <div v-if="isSource && focused" class="toolbar__group">
        <button
          type="button"
          class="toolbar__icon-btn"
          data-tip="撤销 Ctrl+Z"
          aria-label="撤销 (Ctrl+Z)"
          :disabled="!editable || !editor.can().chain().focus().undo().run()"
          @mousedown.prevent
          @click="editor.commands.undo()"
        >
          <EditorIcon name="undo" />
        </button>
        <button
          type="button"
          class="toolbar__icon-btn"
          data-tip="重做 Ctrl+Shift+Z"
          aria-label="重做 (Ctrl+Shift+Z)"
          :disabled="!editable || !editor.can().chain().focus().redo().run()"
          @mousedown.prevent
          @click="editor.commands.redo()"
        >
          <EditorIcon name="redo" />
        </button>
      </div>

      <!-- 右：复制 / 清空 -->
      <div class="toolbar__group toolbar__group--end">
        <!-- 源侧（输入框）：复制 + 清空，两个小图标 -->
        <template v-if="isSource">
          <button
            type="button"
            class="toolbar__icon-btn"
            :class="{ 'is-copied': copied }"
            :data-tip="copied ? '已复制到剪贴板' : '复制中文内容'"
            :aria-label="copied ? '已复制到剪贴板' : '复制中文内容'"
            :disabled="!editor.getText().trim()"
            @mousedown.prevent
            @click="copyContent"
          >
            <EditorIcon :name="copied ? 'check' : 'copy'" />
          </button>
          <button
            type="button"
            class="toolbar__icon-btn toolbar__icon-btn--danger"
            data-tip="清空中文内容"
            aria-label="清空中文内容"
            :disabled="!editable || !editor.getText().trim()"
            @mousedown.prevent
            @click="editor.commands.clearContent()"
          >
            <EditorIcon name="trash" />
          </button>
        </template>

        <!-- 译侧：只留「复制」 -->
        <button
          v-else
          type="button"
          class="toolbar__icon-btn"
          :class="{ 'is-copied': copied }"
          :data-tip="copied ? '已复制到剪贴板' : '复制中文内容'"
          :aria-label="copied ? '已复制到剪贴板' : '复制中文内容'"
          :disabled="!editor.getText().trim()"
          @mousedown.prevent
          @click="copyContent"
        >
          <EditorIcon :name="copied ? 'check' : 'copy'" />
        </button>
      </div>
    </div>
    <EditorContent :editor="editor" />
  </div>
</template>
