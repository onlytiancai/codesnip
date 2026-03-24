<script setup lang="ts">
defineProps<{
  progress: number
  status: string
}>()

const statusText = computed(() => {
  if (props.status === 'completed') return 'Complete!'
  if (props.status === 'failed') return 'Failed'
  if (props.progress < 10) return 'Starting...'
  if (props.progress < 30) return 'Fetching HTML...'
  if (props.progress < 50) return 'Parsing content...'
  if (props.progress < 70) return 'Downloading images...'
  if (props.progress < 90) return 'Converting to markdown...'
  return 'Finalizing...'
})

const props = defineProps<{
  progress: number
  status: string
}>()
</script>

<template>
  <div class="mt-6">
    <div class="flex justify-between items-center mb-2">
      <span class="text-sm font-medium text-gray-700">{{ statusText }}</span>
      <span class="text-sm text-gray-500">{{ progress }}%</span>
    </div>
    <div class="w-full bg-gray-200 rounded-full h-2">
      <div
        :class="[
          'h-2 rounded-full transition-all duration-300',
          status === 'failed' ? 'bg-red-500' : 'bg-blue-500'
        ]"
        :style="{ width: `${progress}%` }"
      />
    </div>
  </div>
</template>
