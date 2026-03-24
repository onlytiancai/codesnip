<script setup lang="ts">
const theme = useState<'light' | 'dark' | 'sepia'>('theme', () => {
  if (import.meta.client) {
    return (localStorage.getItem('theme') as 'light' | 'dark' | 'sepia') || 'light'
  }
  return 'light'
})

watch(theme, (newTheme) => {
  if (import.meta.client) {
    localStorage.setItem('theme', newTheme)
  }
})

const themes = [
  { value: 'light', label: 'Light', icon: '☀️' },
  { value: 'dark', label: 'Dark', icon: '🌙' },
  { value: 'sepia', label: 'Sepia', icon: '📜' }
] as const
</script>

<template>
  <div class="flex items-center gap-2">
    <span class="text-sm text-gray-600">Theme:</span>
    <div class="flex gap-1 bg-gray-100 rounded-lg p-1">
      <button
        v-for="t in themes"
        :key="t.value"
        @click="theme = t.value"
        :title="t.label"
        :class="[
          'px-3 py-1 text-sm rounded-md transition-colors',
          theme === t.value
            ? 'bg-white shadow text-gray-900'
            : 'text-gray-600 hover:text-gray-900'
        ]"
      >
        {{ t.icon }}
      </button>
    </div>
  </div>
</template>
