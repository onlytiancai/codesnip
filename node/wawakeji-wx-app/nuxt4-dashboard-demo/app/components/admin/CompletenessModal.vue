<template>
  <UModal
    v-model:open="isOpen"
    title="Incomplete Content"
    description="The following items are missing required fields. Do you want to save anyway?"
    :ui="{ width: 'sm:max-w-lg' }"
  >
    <template #body>
      <div class="max-h-60 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-lg">
        <ul class="divide-y divide-gray-200 dark:divide-gray-700">
          <li
            v-for="(item, index) in incompleteItems"
            :key="index"
            class="p-3 text-sm"
          >
            <div class="flex items-center gap-2">
              <UIcon name="i-lucide-alert-circle" class="w-4 h-4 text-warning flex-shrink-0" />
              <span class="font-medium">{{ item.type }} #{{ item.order + 1 }}</span>
            </div>
            <p class="text-gray-500 dark:text-gray-400 ml-6 mt-1">
              Missing: {{ item.missing.join(', ') }}
            </p>
            <p v-if="item.preview" class="text-xs text-gray-400 ml-6 mt-1 truncate">
              "{{ item.preview }}"
            </p>
          </li>
        </ul>
      </div>
    </template>

    <template #footer>
      <div class="flex justify-end gap-3">
        <UButton variant="outline" @click="close">
          Go Back
        </UButton>
        <UButton color="warning" @click="saveAnyway">
          Save Anyway
        </UButton>
      </div>
    </template>
  </UModal>
</template>

<script setup lang="ts">
export interface IncompleteItem {
  type: 'Paragraph' | 'Sentence'
  order: number
  missing: string[]
  preview?: string
}

const props = defineProps<{
  open: boolean
  items: IncompleteItem[]
}>()

const emit = defineEmits<{
  'update:open': [value: boolean]
  'save-anyway': []
}>()

const isOpen = computed({
  get: () => props.open,
  set: (value) => emit('update:open', value)
})

const incompleteItems = computed(() => props.items)

const close = () => {
  emit('update:open', false)
}

const saveAnyway = () => {
  emit('save-anyway')
  close()
}
</script>