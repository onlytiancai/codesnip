<script setup>
import { ref, computed } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

const props = defineProps({
  step: String,
  content: String,
  filePath: String
})

const emit = defineEmits(['confirm'])

const isEditing = ref(false)
const editedContent = ref('')
const isApproved = ref(true)

const stepLabels = {
  analyze: 'Content Analysis',
  prompt_gen: 'Translation Prompt',
  review: 'Translation Review'
}

const stepLabel = computed(() => stepLabels[props.step] || props.step)

const renderedContent = computed(() => {
  if (!props.content) return ''
  const rawHtml = marked(props.content)
  return DOMPurify.sanitize(rawHtml)
})

function startEdit() {
  editedContent.value = props.content
  isEditing.value = true
}

function cancelEdit() {
  isEditing.value = false
  editedContent.value = ''
}

function confirmEdit() {
  emit('confirm', isApproved.value, editedContent.value || props.content)
  isEditing.value = false
}

function confirmWithoutEdit() {
  emit('confirm', true, null)
}

function reject() {
  emit('confirm', false, null)
}
</script>

<template>
  <div class="bg-white shadow rounded-lg p-6 mt-6">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-lg font-semibold text-gray-900">
        Confirm: {{ stepLabel }}
      </h2>
      <span v-if="filePath" class="text-sm text-gray-500">
        {{ filePath }}
      </span>
    </div>

    <!-- View Mode -->
    <div v-if="!isEditing" class="space-y-4">
      <div class="prose max-w-none markdown-preview border rounded-lg p-4 max-h-96 overflow-y-auto" v-html="renderedContent"></div>

      <div class="flex items-center justify-between">
        <button
          @click="startEdit"
          class="text-primary-600 hover:text-primary-700 text-sm font-medium"
        >
          Edit before confirming
        </button>

        <div class="flex space-x-3">
          <button
            @click="reject"
            class="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
          >
            Reject
          </button>
          <button
            @click="confirmWithoutEdit"
            class="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
          >
            Confirm
          </button>
        </div>
      </div>
    </div>

    <!-- Edit Mode -->
    <div v-else class="space-y-4">
      <textarea
        v-model="editedContent"
        rows="15"
        class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500 font-mono text-sm"
      ></textarea>

      <div class="flex items-center justify-between">
        <label class="flex items-center space-x-2">
          <input
            v-model="isApproved"
            type="checkbox"
            class="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
          />
          <span class="text-sm text-gray-700">Mark as approved</span>
        </label>

        <div class="flex space-x-3">
          <button
            @click="cancelEdit"
            class="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            @click="confirmEdit"
            class="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
          >
            Save & Confirm
          </button>
        </div>
      </div>
    </div>
  </div>
</template>