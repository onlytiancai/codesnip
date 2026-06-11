<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import EditModal from '@/components/EditModal.vue'

const props = defineProps({
  original: String,
  translation: String,
  files: Array
})

const emit = defineEmits(['save'])

const activeTab = ref('side-by-side')
const showEditModal = ref(false)
const editingFile = ref(null)
const editingContent = ref('')

const tabs = [
  { key: 'side-by-side', label: 'Side by Side' },
  { key: 'original', label: 'Original' },
  { key: 'translation', label: 'Translation' }
]

const renderedOriginal = computed(() => {
  if (!props.original) return ''
  const rawHtml = marked(props.original)
  return DOMPurify.sanitize(rawHtml)
})

const renderedTranslation = computed(() => {
  if (!props.translation) return ''
  const rawHtml = marked(props.translation)
  return DOMPurify.sanitize(rawHtml)
})

const translationFiles = computed(() => {
  return props.files?.filter(f =>
    f.endsWith('.md') &&
    !f.startsWith('04-segments') &&
    !f.startsWith('05-translations')
  ) || []
})

function openEditModal(filePath) {
  editingFile.value = filePath
  // Find the content for this file
  if (filePath === 'translation.md') {
    editingContent.value = props.translation
  } else if (filePath === 'original.md') {
    editingContent.value = props.original
  }
  showEditModal.value = true
}

async function handleSaveEdit(content) {
  await emit('save', editingFile.value, content)
  showEditModal.value = false
}

function getFileLabel(filePath) {
  const labels = {
    'original.md': 'Original',
    '01-analysis.md': 'Analysis',
    '02-terminology.md': 'Terminology',
    '03-prompt.md': 'Prompt',
    '06-draft.md': 'Draft',
    '07-critique.md': 'Critique',
    '08-revision.md': 'Revision',
    'translation.md': 'Final Translation'
  }
  return labels[filePath] || filePath
}
</script>

<template>
  <div class="bg-white shadow rounded-lg">
    <!-- Tabs -->
    <div class="border-b border-gray-200">
      <div class="flex">
        <button
          v-for="tab in tabs"
          :key="tab.key"
          @click="activeTab = tab.key"
          :class="activeTab === tab.key ? 'border-primary-500 text-primary-600' : 'border-transparent text-gray-500 hover:text-gray-700'"
          class="px-4 py-3 border-b-2 font-medium text-sm"
        >
          {{ tab.label }}
        </button>
      </div>
    </div>

    <!-- Side by Side View -->
    <div v-if="activeTab === 'side-by-side'" class="grid grid-cols-2 gap-0 divide-x divide-gray-200">
      <div class="p-4">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-medium text-gray-700">Original</h3>
          <button
            @click="openEditModal('original.md')"
            class="text-xs text-primary-600 hover:text-primary-700"
          >
            Edit
          </button>
        </div>
        <div class="prose max-w-none markdown-preview max-h-[600px] overflow-y-auto text-sm" v-html="renderedOriginal"></div>
      </div>
      <div class="p-4">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-medium text-gray-700">Translation</h3>
          <button
            @click="openEditModal('translation.md')"
            class="text-xs text-primary-600 hover:text-primary-700"
          >
            Edit
          </button>
        </div>
        <div class="prose max-w-none markdown-preview max-h-[600px] overflow-y-auto text-sm" v-html="renderedTranslation"></div>
      </div>
    </div>

    <!-- Original Only -->
    <div v-else-if="activeTab === 'original'" class="p-4">
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-sm font-medium text-gray-700">Original Content</h3>
        <button
          @click="openEditModal('original.md')"
          class="text-xs text-primary-600 hover:text-primary-700"
        >
          Edit
        </button>
      </div>
      <div class="prose max-w-none markdown-preview max-h-[600px] overflow-y-auto" v-html="renderedOriginal"></div>
    </div>

    <!-- Translation Only -->
    <div v-else-if="activeTab === 'translation'" class="p-4">
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-sm font-medium text-gray-700">Translation</h3>
        <button
          @click="openEditModal('translation.md')"
          class="text-xs text-primary-600 hover:text-primary-700"
        >
          Edit
        </button>
      </div>
      <div class="prose max-w-none markdown-preview max-h-[600px] overflow-y-auto" v-html="renderedTranslation"></div>
    </div>

    <!-- Files List -->
    <div v-if="translationFiles.length > 0" class="border-t border-gray-200 p-4">
      <h3 class="text-sm font-medium text-gray-700 mb-2">Project Files</h3>
      <div class="flex flex-wrap gap-2">
        <button
          v-for="file in translationFiles"
          :key="file"
          @click="openEditModal(file)"
          class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-700 hover:bg-gray-200"
        >
          {{ getFileLabel(file) }}
        </button>
      </div>
    </div>

    <!-- Edit Modal -->
    <EditModal
      v-if="showEditModal"
      :file-path="editingFile"
      :content="editingContent"
      @save="handleSaveEdit"
      @close="showEditModal = false"
    />
  </div>
</template>