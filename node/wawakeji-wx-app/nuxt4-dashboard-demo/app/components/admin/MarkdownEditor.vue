<template>
  <div class="markdown-editor">
    <!-- Tab switcher -->
    <div class="flex border-b border-gray-200 dark:border-gray-700 mb-3">
      <button
        type="button"
        :class="[
          'px-4 py-2 text-sm font-medium transition-colors',
          mode === 'edit'
            ? 'text-green-600 border-b-2 border-green-600'
            : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
        ]"
        @click="mode = 'edit'"
      >
        <UIcon name="i-lucide-edit-3" class="w-4 h-4 inline mr-1" />
        Edit
      </button>
      <button
        type="button"
        :class="[
          'px-4 py-2 text-sm font-medium transition-colors',
          mode === 'preview'
            ? 'text-green-600 border-b-2 border-green-600'
            : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
        ]"
        @click="mode = 'preview'"
      >
        <UIcon name="i-lucide-eye" class="w-4 h-4 inline mr-1" />
        Preview
      </button>
      <button
        type="button"
        :class="[
          'px-4 py-2 text-sm font-medium transition-colors',
          mode === 'split'
            ? 'text-green-600 border-b-2 border-green-600'
            : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
        ]"
        @click="mode = 'split'"
      >
        <UIcon name="i-lucide-columns" class="w-4 h-4 inline mr-1" />
        Split
      </button>
    </div>

    <!-- Editor and Preview -->
    <div :class="mode === 'split' ? 'grid grid-cols-2 gap-4' : ''">
      <!-- Editor -->
      <div v-show="mode === 'edit' || mode === 'split'">
        <textarea
          :value="modelValue"
          :placeholder="placeholder"
          :rows="rows"
          :class="[
            'w-full rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 p-3 text-sm font-mono resize-y focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent',
            mode === 'split' ? 'min-h-[400px]' : ''
          ]"
          @input="$emit('update:modelValue', ($event.target as HTMLTextAreaElement).value)"
        />
      </div>

      <!-- Preview -->
      <div
        v-show="mode === 'preview' || mode === 'split'"
        :class="[
          'prose prose-sm dark:prose-invert max-w-none overflow-auto bg-gray-50 dark:bg-gray-900 rounded-md p-4',
          mode === 'split' ? 'min-h-[400px] max-h-[600px]' : 'min-h-[300px] max-h-[600px]'
        ]"
        v-html="renderedContent"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { marked } from 'marked'

const props = defineProps<{
  modelValue: string
  placeholder?: string
  rows?: number
}>()

defineEmits<{
  'update:modelValue': [value: string]
}>()

const mode = ref<'edit' | 'preview' | 'split'>('edit')

// Configure marked options
marked.setOptions({
  breaks: true,
  gfm: true
})

const renderedContent = computed(() => {
  if (!props.modelValue) {
    return '<p class="text-gray-400 italic">No content to preview</p>'
  }
  try {
    return marked.parse(props.modelValue) as string
  } catch (e) {
    return '<p class="text-red-500">Error rendering markdown</p>'
  }
})
</script>

<style scoped>
.markdown-editor :deep(.prose img) {
  max-width: 100%;
  height: auto;
  border-radius: 0.5rem;
}

.markdown-editor :deep(.prose pre) {
  background-color: #f3f4f6;
  padding: 1rem;
  border-radius: 0.5rem;
  overflow-x: auto;
}

:global(.dark) .markdown-editor :deep(.prose pre) {
  background-color: #1f2937;
}

.markdown-editor :deep(.prose code) {
  background-color: #f3f4f6;
  padding: 0.125rem 0.25rem;
  border-radius: 0.25rem;
  font-size: 0.875em;
}

:global(.dark) .markdown-editor :deep(.prose code) {
  background-color: #1f2937;
}

.markdown-editor :deep(.prose pre code) {
  background-color: transparent;
  padding: 0;
}

.markdown-editor :deep(.prose table) {
  width: 100%;
  border-collapse: collapse;
}

.markdown-editor :deep(.prose th),
.markdown-editor :deep(.prose td) {
  border: 1px solid #d1d5db;
  padding: 0.5rem;
}

:global(.dark) .markdown-editor :deep(.prose th),
:global(.dark) .markdown-editor :deep(.prose td) {
  border-color: #4b5563;
}

.markdown-editor :deep(.prose blockquote) {
  border-left: 4px solid #22c55e;
  padding-left: 1rem;
  margin-left: 0;
  color: #4b5563;
}
</style>