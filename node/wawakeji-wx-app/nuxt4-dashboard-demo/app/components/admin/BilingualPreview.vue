<template>
  <div class="bilingual-preview">
    <!-- View Mode Tabs -->
    <div class="flex items-center gap-2 mb-4 border-b pb-4">
      <UButton
        v-for="mode in viewModes"
        :key="mode.value"
        :color="viewMode === mode.value ? 'primary' : 'neutral'"
        :variant="viewMode === mode.value ? 'solid' : 'ghost'"
        size="sm"
        @click="viewMode = mode.value"
      >
        {{ mode.label }}
      </UButton>
    </div>

    <!-- Content Display using normal markdown rendering -->
    <article class="markdown-body bg-white dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
      <!-- Original Only Mode -->
      <template v-if="viewMode === 'original'">
        <div v-html="renderedOriginal" />
      </template>

      <!-- Translated Only Mode -->
      <template v-else-if="viewMode === 'translated'">
        <div v-html="renderedTranslated" />
      </template>

      <!-- Bilingual Mode (Immersive: EN paragraph, then CN translation) -->
      <template v-else>
        <template v-for="(block, index) in blocks" :key="index">
          <!-- Heading: render both EN and CN -->
          <template v-if="block.type === 'heading'">
            <component :is="`h${block.level}`" class="mt-6 mb-3">
              <span v-html="renderInlineMarkdown(block.original)" />
            </component>
            <component :is="`h${block.level}`" class="mt-1 mb-4 text-gray-600 dark:text-gray-400 font-normal">
              <span v-html="renderInlineMarkdown(block.translated)" />
            </component>
          </template>

          <!-- Code Block: only show original (no translation) -->
          <template v-else-if="block.type === 'code'">
            <pre><code :class="block.language ? `language-${block.language}` : ''">{{ block.original }}</code></pre>
          </template>

          <!-- Image: only show original (no translation) -->
          <template v-else-if="block.type === 'image'">
            <div v-html="renderMarkdown(block.original)" />
          </template>

          <!-- Horizontal Rule -->
          <template v-else-if="block.type === 'hr'">
            <hr class="my-6 border-gray-300 dark:border-gray-600" />
          </template>

          <!-- Table: show EN then CN -->
          <template v-else-if="block.type === 'table'">
            <div v-html="renderMarkdown(block.original)" />
            <div class="mt-4 text-sm text-gray-500 mb-2">中文翻译:</div>
            <div v-html="renderMarkdown(block.translated)" />
          </template>

          <!-- Blockquote: EN then CN -->
          <template v-else-if="block.type === 'blockquote'">
            <blockquote class="border-l-4 border-gray-300 pl-4 italic my-4">
              <div v-html="renderMarkdown(block.original)" />
            </blockquote>
            <blockquote class="border-l-4 border-primary-400 pl-4 italic my-4 text-gray-700 dark:text-gray-300">
              <div v-html="renderMarkdown(block.translated)" />
            </blockquote>
          </template>

          <!-- List: EN then CN -->
          <template v-else-if="block.type === 'list'">
            <div v-html="renderMarkdown(block.original)" />
            <div class="mt-2 text-gray-700 dark:text-gray-300" v-html="renderMarkdown(block.translated)" />
          </template>

          <!-- Paragraph: EN then CN (immersive translation style) -->
          <template v-else>
            <p class="mb-2" v-html="renderInlineMarkdown(block.original)" />
            <p class="text-gray-700 dark:text-gray-300 mb-4" v-html="renderInlineMarkdown(block.translated)" />
          </template>
        </template>
      </template>
    </article>
  </div>
</template>

<script setup lang="ts">
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import 'github-markdown-css/github-markdown.css'

interface BilingualBlock {
  original: string
  translated: string
  type: string
  level?: number
  language?: string
}

const props = defineProps<{
  blocks: BilingualBlock[]
}>()

const viewMode = ref<'original' | 'translated' | 'both'>('both')

const viewModes = [
  { label: 'Bilingual', value: 'both' },
  { label: 'Original', value: 'original' },
  { label: 'Translated', value: 'translated' }
]

// Configure marked for better markdown rendering
marked.setOptions({
  breaks: true,
  gfm: true
})

const renderMarkdown = (text: string): string => {
  if (!text) return ''
  try {
    const html = marked.parse(text) as string
    return DOMPurify.sanitize(html)
  } catch {
    return text
  }
}

const renderInlineMarkdown = (text: string): string => {
  if (!text) return ''
  try {
    // Parse as inline markdown (no paragraph wrapping)
    const html = marked.parseInline(text) as string
    return DOMPurify.sanitize(html)
  } catch {
    return text
  }
}

const renderedOriginal = computed(() => {
  const markdown = props.blocks
    .map((block) => {
      switch (block.type) {
        case 'code':
          return `\`\`\`${block.language || ''}\n${block.original}\n\`\`\``
        case 'hr':
          return '---'
        case 'image':
          return block.original
        case 'heading':
          return `${'#'.repeat(block.level || 1)} ${block.original}`
        default:
          return block.original
      }
    })
    .join('\n\n')
  return renderMarkdown(markdown)
})

const renderedTranslated = computed(() => {
  const markdown = props.blocks
    .map((block) => {
      switch (block.type) {
        case 'code':
          return `\`\`\`${block.language || ''}\n${block.original}\n\`\`\``
        case 'hr':
          return '---'
        case 'image':
          return block.original
        case 'heading':
          return `${'#'.repeat(block.level || 1)} ${block.translated}`
        default:
          return block.translated
      }
    })
    .join('\n\n')
  return renderMarkdown(markdown)
})
</script>

<style scoped>
/* GitHub markdown CSS already provides base styling */
/* Only add custom styles for bilingual-specific elements */

.bilingual-preview :deep(.markdown-body) {
  font-size: 1rem;
  line-height: 1.75;
  max-width: 100%;
}

/* Dark mode adjustments - github-markdown-css uses light theme by default */
.dark .bilingual-preview :deep(.markdown-body) {
  background-color: transparent;
  color: #e5e7eb;
}

.dark .bilingual-preview :deep(.markdown-body pre) {
  background: #1f2937;
}

.dark .bilingual-preview :deep(.markdown-body code) {
  background: #374151;
}

.dark .bilingual-preview :deep(.markdown-body table th) {
  background: #1f2937;
}

.dark .bilingual-preview :deep(.markdown-body table td),
.dark .bilingual-preview :deep(.markdown-body table th) {
  border-color: #374151;
}

.dark .bilingual-preview :deep(.markdown-body blockquote) {
  color: #9ca3af;
  border-left-color: #4b5563;
}

/* Custom bilingual segment styling */
.bilingual-preview :deep(h1),
.bilingual-preview :deep(h2),
.bilingual-preview :deep(h3),
.bilingual-preview :deep(h4),
.bilingual-preview :deep(h5),
.bilingual-preview :deep(h6) {
  margin-top: 1.5rem;
  margin-bottom: 0.75rem;
}

/* Translated heading styling */
.bilingual-preview :deep(h1.font-normal),
.bilingual-preview :deep(h2.font-normal),
.bilingual-preview :deep(h3.font-normal),
.bilingual-preview :deep(h4.font-normal) {
  opacity: 0.8;
}

/* Blockquote styling for bilingual */
.bilingual-preview :deep(blockquote.border-primary-400) {
  border-left-color: #3b82f6;
}
</style>