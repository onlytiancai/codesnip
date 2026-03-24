<script setup lang="ts">
defineProps<{
  content: string
}>()

const theme = useState<'light' | 'dark' | 'sepia'>('theme', () => {
  if (import.meta.client) {
    return (localStorage.getItem('theme') as 'light' | 'dark' | 'sepia') || 'light'
  }
  return 'light'
})

const themeClasses = computed(() => {
  switch (theme.value) {
    case 'dark':
      return 'bg-gray-900 text-gray-100'
    case 'sepia':
      return 'bg-amber-50 text-gray-800'
    default:
      return 'bg-white text-gray-900'
  }
})
</script>

<template>
  <div :class="['prose max-w-none p-8 rounded-lg', themeClasses]" v-html="content" />
</template>

<style>
.prose {
  font-family: system-ui, -apple-system, sans-serif;
  line-height: 1.75;
}

.prose h1 { font-size: 2em; font-weight: bold; margin: 0.67em 0; }
.prose h2 { font-size: 1.5em; font-weight: bold; margin: 0.83em 0; }
.prose h3 { font-size: 1.17em; font-weight: bold; margin: 1em 0; }
.prose p { margin: 1em 0; }
.prose img { max-width: 100%; height: auto; margin: 1em 0; }
.prose a { color: #3b82f6; text-decoration: underline; }
.prose blockquote { border-left: 4px solid #d1d5db; padding-left: 1em; margin: 1em 0; font-style: italic; }
.prose code { background: #f3f4f6; padding: 0.2em 0.4em; border-radius: 3px; font-size: 0.875em; }
.prose pre { background: #1f2937; color: #f3f4f6; padding: 1em; border-radius: 0.5em; overflow-x: auto; margin: 1em 0; }
.prose pre code { background: transparent; padding: 0; }
.prose ul, .prose ol { margin: 1em 0; padding-left: 2em; }
.prose li { margin: 0.5em 0; }
.prose table { border-collapse: collapse; margin: 1em 0; width: 100%; }
.prose th, .prose td { border: 1px solid #d1d5db; padding: 0.5em; text-align: left; }
.prose th { background: #f3f4f6; font-weight: bold; }
</style>
