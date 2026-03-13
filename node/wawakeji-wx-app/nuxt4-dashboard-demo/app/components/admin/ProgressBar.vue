<template>
  <div v-if="show" class="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 w-full max-w-md mx-4">
      <div class="flex items-center justify-between mb-4">
        <h3 class="font-semibold text-lg">{{ title }}</h3>
        <UButton
          v-if="cancellable"
          variant="ghost"
          icon="i-lucide-x"
          size="sm"
          @click="$emit('cancel')"
        />
      </div>

      <div class="space-y-3">
        <!-- Progress Bar -->
        <div class="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            class="h-full bg-primary transition-all duration-300 rounded-full"
            :style="{ width: progressPercent + '%' }"
          />
        </div>

        <!-- Progress Text -->
        <div class="flex justify-between text-sm text-gray-500 dark:text-gray-400">
          <span>{{ currentItem }}</span>
          <span>{{ current }} / {{ total }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  show: boolean
  title: string
  current: number
  total: number
  currentItem?: string
  cancellable?: boolean
}>()

defineEmits<{
  cancel: []
}>()

const progressPercent = computed(() => {
  if (props.total === 0) return 0
  return Math.round((props.current / props.total) * 100)
})
</script>