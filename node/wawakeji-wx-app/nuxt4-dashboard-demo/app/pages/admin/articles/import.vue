<template>
  <NuxtLayout name="admin">
    <div class="w-full">
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-2xl font-bold">Import Article from URL</h2>
        <UButton variant="ghost" to="/admin/articles">
          <UIcon name="i-lucide-arrow-left" class="w-4 h-4 mr-2" />
          Back to List
        </UButton>
      </div>

      <!-- URL Input -->
      <UCard class="mb-6">
        <div class="flex gap-3">
          <UInput
            v-model="url"
            placeholder="Enter article URL (e.g., https://example.com/article)"
            class="flex-1"
            :disabled="isProcessing"
            @keyup.enter="startImport"
          />
          <UButton
            color="primary"
            :loading="isCreating"
            :disabled="!url || isProcessing"
            @click="startImport"
          >
            Fetch
          </UButton>
          <UButton
            v-if="isProcessing"
            color="error"
            variant="outline"
            :loading="isCancelling"
            @click="cancelImport"
          >
            Cancel
          </UButton>
        </div>
      </UCard>

      <!-- Progress State -->
      <UCard v-if="taskId && !preview && !error" class="mb-6">
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <h3 class="font-semibold">Processing Article</h3>
            <UBadge :color="stageBadgeColor">{{ stageLabel }}</UBadge>
          </div>

          <UProgress :value="progress" :max="100" />

          <div class="flex items-center justify-between text-sm text-gray-500">
            <span>{{ progressMessage }}</span>
            <span>{{ progress }}%</span>
          </div>

          <!-- Image download progress -->
          <div v-if="stage === 'images' && totalImages > 0" class="text-sm text-gray-500">
            <UIcon name="i-lucide-image" class="w-4 h-4 inline mr-1" />
            Downloading images: {{ processedImages }} / {{ totalImages }}
          </div>
        </div>
      </UCard>

      <!-- Preview -->
      <template v-else-if="preview">
        <UCard class="mb-6">
          <template #header>
            <div class="flex items-center justify-between">
              <h3 class="font-semibold">Extracted Content</h3>
              <div class="flex items-center gap-2">
                <UBadge v-if="preview.images?.length" color="primary" variant="subtle">
                  {{ preview.images.length }} image(s) downloaded
                </UBadge>
                <a :href="preview.url" target="_blank" class="text-sm text-primary hover:underline">
                  View Original
                </a>
              </div>
            </div>
          </template>

          <div class="space-y-4">
            <!-- Title -->
            <UFormField label="Title" name="title" required>
              <UInput v-model="preview.title" placeholder="Article title" class="w-full" />
            </UFormField>

            <!-- Content -->
            <UFormField label="Content (Markdown)" name="content" required>
              <UTextarea
                v-model="preview.markdown"
                placeholder="Article content in Markdown format..."
                :rows="20"
                class="w-full font-mono text-sm"
              />
            </UFormField>
          </div>
        </UCard>

        <!-- Actions -->
        <div class="flex justify-end gap-3">
          <UButton variant="outline" to="/admin/articles">
            Cancel
          </UButton>
          <UButton
            color="primary"
            icon="i-lucide-arrow-right"
            :loading="proceeding"
            @click="proceedToCreate"
          >
            Use This Content
          </UButton>
        </div>
      </template>

      <!-- Error State -->
      <UCard v-else-if="error" class="bg-red-50 dark:bg-red-950">
        <div class="flex items-start gap-3">
          <UIcon name="i-lucide-alert-circle" class="w-5 h-5 text-red-500 mt-0.5" />
          <div>
            <p class="font-medium text-red-700 dark:text-red-300">Failed to fetch article</p>
            <p class="text-sm text-red-600 dark:text-red-400 mt-1">{{ error }}</p>
          </div>
        </div>
      </UCard>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false,
  middleware: 'admin'
})

interface ImportResult {
  title: string
  markdown: string
  url: string
  images: Array<{ originalUrl: string; localUrl: string }>
}

interface TaskUpdate {
  id: number
  status: string
  stage: string
  progress: number
  totalImages: number
  processedImages: number
  result?: ImportResult
  error?: string
}

const url = ref('')
const taskId = ref<number | null>(null)
const status = ref('')
const stage = ref('')
const progress = ref(0)
const totalImages = ref(0)
const processedImages = ref(0)
const preview = ref<ImportResult | null>(null)
const error = ref<string | null>(null)
const proceeding = ref(false)
const isCreating = ref(false)
const isCancelling = ref(false)
const toast = useToast()

// EventSource reference for cleanup
let eventSource: EventSource | null = null

const isProcessing = computed(() => {
  return taskId.value && !preview.value && !error.value
})

const stageLabel = computed(() => {
  const labels: Record<string, string> = {
    init: 'Initializing',
    fetch: 'Fetching',
    extract: 'Extracting',
    convert: 'Converting',
    images: 'Downloading Images',
    done: 'Complete'
  }
  return labels[stage.value] || 'Processing'
})

const stageBadgeColor = computed(() => {
  if (status.value === 'failed') return 'error'
  if (status.value === 'completed') return 'success'
  return 'primary'
})

const progressMessage = computed(() => {
  switch (stage.value) {
    case 'fetch':
      return 'Fetching webpage content...'
    case 'extract':
      return 'Extracting article content...'
    case 'convert':
      return 'Converting to Markdown...'
    case 'images':
      return 'Downloading images...'
    case 'done':
      return 'Complete!'
    default:
      return 'Starting...'
  }
})

const startImport = async () => {
  if (!url.value) return

  // Reset state
  error.value = null
  preview.value = null
  taskId.value = null
  stage.value = ''
  progress.value = 0
  status.value = ''

  // Validate URL
  try {
    new URL(url.value)
  } catch {
    error.value = 'Please enter a valid URL'
    return
  }

  isCreating.value = true

  try {
    // Create import task
    const result = await $fetch('/api/admin/articles/import-queue', {
      method: 'POST',
      body: { url: url.value }
    })

    taskId.value = result.taskId
    status.value = result.status
    stage.value = result.stage

    // Subscribe to SSE updates
    subscribeToUpdates(result.taskId)
  } catch (e: any) {
    error.value = e.data?.message || e.message || 'Failed to create import task'
    toast.add({
      title: 'Failed to start import',
      description: error.value,
      color: 'error'
    })
  } finally {
    isCreating.value = false
  }
}

const subscribeToUpdates = (id: number) => {
  // Close existing connection if any
  if (eventSource) {
    eventSource.close()
  }

  eventSource = new EventSource(`/api/admin/articles/import-queue/${id}/events`)

  eventSource.onmessage = (event) => {
    const data: TaskUpdate = JSON.parse(event.data)

    status.value = data.status
    stage.value = data.stage
    progress.value = data.progress
    totalImages.value = data.totalImages || 0
    processedImages.value = data.processedImages || 0

    if (data.status === 'completed' && data.result) {
      preview.value = data.result

      if (data.result.images?.length > 0) {
        toast.add({
          title: `${data.result.images.length} image(s) downloaded`,
          color: 'success'
        })
      }

      eventSource?.close()
      eventSource = null
    } else if (data.status === 'failed') {
      error.value = data.error || 'Import failed'
      toast.add({
        title: 'Import failed',
        description: error.value,
        color: 'error'
      })
      eventSource?.close()
      eventSource = null
    } else if (data.status === 'cancelled') {
      error.value = 'Import was cancelled'
      eventSource?.close()
      eventSource = null
    }
  }

  eventSource.onerror = () => {
    console.error('SSE connection error')
    // Fall back to polling
    pollForUpdates(id)
  }
}

const pollForUpdates = async (id: number) => {
  const poll = async () => {
    if (!taskId.value || preview.value || error.value) return

    try {
      const data = await $fetch(`/api/admin/articles/import-queue/${id}`)

      status.value = data.status
      stage.value = data.stage
      progress.value = data.progress
      totalImages.value = data.totalImages || 0
      processedImages.value = data.processedImages || 0

      if (data.status === 'completed' && data.result) {
        preview.value = data.result

        if (data.result.images?.length > 0) {
          toast.add({
            title: `${data.result.images.length} image(s) downloaded`,
            color: 'success'
          })
        }
      } else if (data.status === 'failed') {
        error.value = data.error || 'Import failed'
        toast.add({
          title: 'Import failed',
          description: error.value,
          color: 'error'
        })
      } else if (data.status !== 'cancelled') {
        // Continue polling
        setTimeout(poll, 1000)
      }
    } catch (e) {
      console.error('Poll error:', e)
      setTimeout(poll, 2000)
    }
  }

  poll()
}

const cancelImport = async () => {
  if (!taskId.value) return

  isCancelling.value = true

  try {
    await $fetch(`/api/admin/articles/import-queue/${taskId.value}`, {
      method: 'DELETE'
    })

    if (eventSource) {
      eventSource.close()
      eventSource = null
    }

    taskId.value = null
    error.value = 'Import was cancelled'

    toast.add({
      title: 'Import cancelled',
      color: 'warning'
    })
  } catch (e: any) {
    toast.add({
      title: 'Failed to cancel import',
      description: e.data?.message || e.message,
      color: 'error'
    })
  } finally {
    isCancelling.value = false
  }
}

const proceedToCreate = async () => {
  if (!preview.value) return

  proceeding.value = true

  try {
    // Store in sessionStorage
    sessionStorage.setItem('importedArticle', JSON.stringify({
      title: preview.value.title,
      content: preview.value.markdown
    }))

    // Navigate to create page
    await navigateTo('/admin/articles/create')
  } finally {
    proceeding.value = false
  }
}

// Cleanup on unmount
onUnmounted(() => {
  if (eventSource) {
    eventSource.close()
    eventSource = null
  }
})
</script>