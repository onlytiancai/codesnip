<script setup>
import { ref, watch, computed } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

const props = defineProps({
  filePath: String,
  content: String
})

const emit = defineEmits(['save', 'close'])

const editedContent = ref('')
const isPreviewMode = ref(false)

watch(() => props.content, (newContent) => {
  editedContent.value = newContent || ''
}, { immediate: true })

const renderedPreview = computed(() => {
  if (!editedContent.value) return ''
  const rawHtml = marked(editedContent.value)
  return DOMPurify.sanitize(rawHtml)
})

function handleSave() {
  emit('save', editedContent.value)
}

function handleClose() {
  emit('close')
}
</script>

<template>
  <div class="fixed inset-0 z-50 overflow-y-auto">
    <div class="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:p-0">
      <!-- Background overlay -->
      <div class="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" @click="handleClose"></div>

      <!-- Modal panel -->
      <div class="relative inline-block w-full max-w-4xl p-6 my-8 text-left align-middle transition-all transform bg-white shadow-xl rounded-lg">
        <!-- Header -->
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-gray-900">
            Edit: {{ filePath }}
          </h3>
          <div class="flex items-center space-x-2">
            <button
              @click="isPreviewMode = !isPreviewMode"
              class="text-sm text-gray-500 hover:text-gray-700"
            >
              {{ isPreviewMode ? 'Edit' : 'Preview' }}
            </button>
            <button
              @click="handleClose"
              class="text-gray-400 hover:text-gray-500"
            >
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <!-- Content -->
        <div class="mb-4">
          <textarea
            v-if="!isPreviewMode"
            v-model="editedContent"
            rows="20"
            class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500 font-mono text-sm"
          ></textarea>
          <div
            v-else
            class="prose max-w-none markdown-preview border rounded-lg p-4 max-h-[500px] overflow-y-auto"
            v-html="renderedPreview"
          ></div>
        </div>

        <!-- Footer -->
        <div class="flex justify-end space-x-3">
          <button
            @click="handleClose"
            class="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            @click="handleSave"
            class="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  </div>
</template>