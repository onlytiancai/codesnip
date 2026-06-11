<script setup lang="ts">
import { marked } from 'marked'
import { nextTick } from 'vue'
import 'github-markdown-css'

const props = defineProps<{
  content: string
  theme?: 'light' | 'dark' | 'sepia'
  editable?: boolean
}>()

const emit = defineEmits<{
  'update:content': [content: string]
}>()

const theme = useState<'light' | 'dark' | 'sepia'>('theme', () => {
  if (import.meta.client) {
    return (localStorage.getItem('theme') as 'light' | 'dark' | 'sepia') || 'light'
  }
  return 'light'
})

const activeTheme = computed(() => props.theme || theme.value)

const containerClasses = computed(() => {
  switch (activeTheme.value) {
    case 'dark':
      return 'markdown-body-dark'
    case 'sepia':
      return 'markdown-body-sepia'
    default:
      return 'markdown-body'
  }
})

const renderedContent = computed(() => {
  return marked(props.content)
})

const editableDiv = ref<HTMLDivElement | null>(null)

function handleInput() {
  if (editableDiv.value && props.editable) {
    emit('update:content', editableDiv.value.innerHTML)
  }
}

function handlePaste(e: ClipboardEvent) {
  if (!props.editable) return

  e.preventDefault()
  const text = e.clipboardData?.getData('text/plain') || ''
  document.execCommand('insertText', false, text)
}

onMounted(() => {
  if (editableDiv.value && props.editable) {
    editableDiv.value.innerHTML = renderedContent.value
  }
})

watch(() => props.content, () => {
  if (editableDiv.value && props.editable && editableDiv.value.innerHTML !== renderedContent.value) {
    editableDiv.value.innerHTML = renderedContent.value
  }
})

watch(() => props.editable, async (isEditable) => {
  if (isEditable) {
    await nextTick()
    if (editableDiv.value) {
      editableDiv.value.innerHTML = renderedContent.value
    }
  }
})
</script>

<template>
  <div :class="['markdown-body-wrapper', containerClasses]">
    <div
      v-if="editable"
      ref="editableDiv"
      class="markdown-body"
      contenteditable="true"
      @input="handleInput"
      @paste="handlePaste"
    />
    <div v-else class="markdown-body" v-html="renderedContent" />
  </div>
</template>

<style>
.markdown-body-wrapper {
  width: 100%;
  height: 100%;
  overflow: auto;
}

.markdown-body {
  padding: 24px;
  width: 100%;
  min-height: 100%;
  box-sizing: border-box;
}

.markdown-body[contenteditable="true"] {
  outline: none;
  cursor: text;
}

.markdown-body[contenteditable="true"]:focus {
  box-shadow: inset 0 0 0 2px rgba(59, 130, 246, 0.5);
}

/* Light theme (GitHub default) */
.markdown-body-dark {
  background-color: #0d1117;
  color: #c9d1d9;
}

.markdown-body-dark .markdown-body {
  background-color: #161b22;
  color: #c9d1d9;
}

.markdown-body-dark a {
  color: #58a6ff;
}

.markdown-body-dark code {
  background-color: #484f58;
  color: #c9d1d9;
}

.markdown-body-dark pre {
  background-color: #484f58;
  border: 1px solid #30363d;
}

.markdown-body-dark pre code {
  background-color: transparent;
}

.markdown-body-dark blockquote {
  border-left-color: #30363d;
  color: #8b949e;
}

.markdown-body-dark table {
  border-color: #30363d;
}

.markdown-body-dark th, .markdown-body-dark td {
  border-color: #30363d;
}

.markdown-body-dark th {
  background-color: #484f58;
}

.markdown-body-dark tr {
  background-color: #161b22;
}

.markdown-body-dark hr {
  border-color: #30363d;
}

/* Sepia theme */
.markdown-body-sepia {
  background-color: #f4ecd8;
}

.markdown-body-sepia .markdown-body {
  background-color: #faf6eb;
  color: #5c4b37;
}

.markdown-body-sepia a {
  color: #b8860b;
}

.markdown-body-sepia code {
  background-color: #e8dcc8;
  color: #5c4b37;
}

.markdown-body-sepia pre {
  background-color: #e8dcc8;
  border: 1px solid #d4c4a8;
}

.markdown-body-sepia pre code {
  background-color: transparent;
}

.markdown-body-sepia blockquote {
  border-left-color: #d4c4a8;
  color: #8b7355;
}

.markdown-body-sepia table {
  border-color: #d4c4a8;
}

.markdown-body-sepia th, .markdown-body-sepia td {
  border-color: #d4c4a8;
}

.markdown-body-sepia th {
  background-color: #e8dcc8;
}

.markdown-body-sepia tr {
  background-color: #faf6eb;
}

.markdown-body-sepia hr {
  border-color: #d4c4a8;
}

/* Ensure images and links are responsive */
.markdown-body img {
  max-width: 100%;
  height: auto;
}

.markdown-body a {
  text-decoration: none;
}

.markdown-body a:hover {
  text-decoration: underline;
}

/* Fix for zero-width characters in links */
.markdown-body a::before {
  content: '';
}
</style>
