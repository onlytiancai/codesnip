<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import { useTranslationStore } from '@/stores/translation'
import WorkflowProgress from '@/components/WorkflowProgress.vue'
import ConfirmPanel from '@/components/ConfirmPanel.vue'
import PreviewPanel from '@/components/PreviewPanel.vue'

const route = useRoute()
const store = useTranslationStore()

const projectId = route.params.id
const activeTab = ref('preview')

const originalContent = ref('')
const translatedContent = ref('')

const project = computed(() => store.currentProject)
const workflowStatus = computed(() => store.workflowStatus)
const isWaitingConfirmation = computed(() => workflowStatus.value.isWaitingConfirmation)

onMounted(async () => {
  await store.fetchProject(projectId)

  // Load original and translation content
  if (project.value) {
    originalContent.value = await store.getFile('original.md') || ''
    translatedContent.value = await store.getFile('translation.md') || ''

    // If project is in progress or waiting, connect to websocket
    if (['in_progress', 'waiting_confirmation'].includes(project.value.status)) {
      store.connectWebSocket(projectId)
    }
  }
})

onUnmounted(() => {
  store.disconnectWebSocket()
})

async function handleStart() {
  await store.startTranslation(projectId)
}

async function handleResume() {
  await store.resumeTranslation(projectId)
}

async function handleConfirm(approved, modifications) {
  await store.confirmStep(
    workflowStatus.value.confirmationStep,
    approved,
    modifications
  )
}

async function handleSaveEdit(filePath, content) {
  await store.updateFile(filePath, content)
}

function formatDate(date) {
  return new Date(date).toLocaleString()
}
</script>

<template>
  <div>
    <!-- Header -->
    <div class="mb-6">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">
            {{ project?.title || 'Translation Project' }}
          </h1>
          <p class="mt-1 text-sm text-gray-500">
            {{ project?.url }}
          </p>
        </div>
        <div class="flex items-center space-x-3">
          <span
            :class="{
              'bg-green-100 text-green-800': project?.status === 'completed',
              'bg-blue-100 text-blue-800': project?.status === 'in_progress',
              'bg-yellow-100 text-yellow-800': project?.status === 'waiting_confirmation',
              'bg-gray-100 text-gray-800': project?.status === 'pending'
            }"
            class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium"
          >
            {{ project?.status?.replace('_', ' ') }}
          </span>

          <button
            v-if="project?.status === 'pending'"
            @click="handleStart"
            :disabled="store.isLoading"
            class="bg-primary-600 text-white px-4 py-2 rounded-md hover:bg-primary-700 disabled:opacity-50"
          >
            Start Translation
          </button>

          <button
            v-if="project?.status === 'waiting_confirmation'"
            @click="handleResume"
            :disabled="store.isLoading"
            class="bg-primary-600 text-white px-4 py-2 rounded-md hover:bg-primary-700 disabled:opacity-50"
          >
            Resume
          </button>
        </div>
      </div>

      <div class="mt-4 flex items-center text-sm text-gray-500 space-x-4">
        <span>Mode: {{ project?.mode }}</span>
        <span>Target: {{ project?.target_language }}</span>
        <span>Created: {{ formatDate(project?.created_at) }}</span>
      </div>
    </div>

    <!-- Workflow Progress -->
    <WorkflowProgress
      v-if="project?.status !== 'pending'"
      :current-step="project?.current_step"
      :mode="project?.mode"
      :message="workflowStatus.message"
    />

    <!-- Confirmation Panel -->
    <ConfirmPanel
      v-if="isWaitingConfirmation"
      :step="workflowStatus.confirmationStep"
      :content="workflowStatus.confirmationContent"
      :file-path="workflowStatus.confirmationFilePath"
      @confirm="handleConfirm"
    />

    <!-- Preview Panel -->
    <div v-if="project?.status === 'completed' || translatedContent" class="mt-6">
      <PreviewPanel
        :original="originalContent"
        :translation="translatedContent"
        :files="project?.files || []"
        @save="handleSaveEdit"
      />
    </div>

    <!-- Loading State -->
    <div v-if="store.isLoading" class="text-center py-12">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
      <p class="mt-4 text-gray-500">Loading...</p>
    </div>
  </div>
</template>