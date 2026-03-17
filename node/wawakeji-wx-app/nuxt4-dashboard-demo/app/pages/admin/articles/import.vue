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
        </div>
      </UCard>

      <!-- Preview -->
      <template v-else-if="preview">
        <!-- Action Bar -->
        <div class="mb-4 flex items-center gap-3 flex-wrap">
          <!-- Image Status & Fetch Button -->
          <template v-if="hasImages">
            <UBadge v-if="imagesDownloaded" color="success" variant="subtle">
              <UIcon name="i-lucide-check" class="w-3 h-3 mr-1" />
              {{ preview.images?.length || 0 }} image(s) downloaded
            </UBadge>
            <template v-else>
              <UBadge color="warning" variant="subtle">
                <UIcon name="i-lucide-image" class="w-3 h-3 mr-1" />
                {{ preview.imageUrls?.length || 0 }} remote image(s)
              </UBadge>
              <UButton
                color="secondary"
                variant="outline"
                icon="i-lucide-download"
                size="sm"
                :loading="isFetchingImages"
                @click="fetchImages"
              >
                Fetch Images
              </UButton>
            </template>
          </template>

          <!-- Translation Button -->
          <UButton
            v-if="!translatedBlocks && !isTranslating"
            color="secondary"
            icon="i-lucide-languages"
            :disabled="isTranslating"
            @click="startTranslation"
          >
            Translate to Chinese
          </UButton>
          <template v-if="isTranslating">
            <UIcon name="i-lucide-loader-2" class="w-4 h-4 animate-spin" />
            <span class="text-sm text-gray-500">Translating... {{ translateProgress }}</span>
          </template>
          <UButton
            v-if="translatedBlocks"
            color="neutral"
            variant="ghost"
            icon="i-lucide-refresh-cw"
            size="sm"
            @click="startTranslation"
          >
            Re-translate
          </UButton>
        </div>

        <UCard class="mb-6">
          <template #header>
            <div class="flex items-center justify-between">
              <h3 class="font-semibold">Extracted Content</h3>
              <a :href="preview.url" target="_blank" class="text-sm text-primary hover:underline">
                View Original
              </a>
            </div>
          </template>

          <div class="space-y-4">
            <!-- Title -->
            <UFormField label="Title" name="title" required>
              <UInput v-model="preview.title" placeholder="Article title" class="w-full" />
            </UFormField>

            <!-- Translated Title -->
            <UFormField v-if="translatedTitle" label="Translated Title" name="translatedTitle">
              <UInput v-model="translatedTitle" placeholder="Translated title" class="w-full" />
            </UFormField>

            <!-- Content Mode Tabs -->
            <div class="flex items-center gap-2 border-b pb-2 flex-wrap">
              <UButton
                v-for="mode in contentModes"
                :key="mode.value"
                :color="contentMode === mode.value ? 'primary' : 'neutral'"
                :variant="contentMode === mode.value ? 'solid' : 'ghost'"
                size="xs"
                @click="contentMode = mode.value"
              >
                {{ mode.label }}
              </UButton>
            </div>

            <!-- Original HTML Preview -->
            <article v-if="contentMode === 'original-md'" class="markdown-body bg-white dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <div v-html="renderedOriginalMarkdown" />
            </article>

            <!-- Edit Original Markdown -->
            <UFormField v-else-if="contentMode === 'edit'" label="Content (Markdown)" name="content" required>
              <UTextarea
                v-model="preview.markdown"
                placeholder="Article content in Markdown format..."
                :rows="20"
                class="w-full font-mono text-sm"
              />
            </UFormField>

            <!-- Bilingual Content -->
            <UFormField v-else-if="contentMode === 'bilingual'" label="Bilingual Content (Markdown)" name="bilingualContent">
              <UTextarea
                v-model="bilingualMarkdown"
                placeholder="Bilingual content in Markdown format..."
                :rows="20"
                class="w-full font-mono text-sm"
              />
            </UFormField>

            <!-- Bilingual Preview -->
            <template v-else-if="contentMode === 'preview-bilingual' && translatedBlocks">
              <AdminBilingualPreview :blocks="translatedBlocks" />
            </template>

            <!-- Placeholder when no translation yet -->
            <div v-else-if="contentMode === 'preview-bilingual' && !translatedBlocks" class="text-center py-8 text-gray-500">
              <UIcon name="i-lucide-languages" class="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>Click "Translate to Chinese" to generate bilingual preview</p>
            </div>
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
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import 'github-markdown-css/github-markdown.css'

definePageMeta({
  layout: false,
  middleware: 'admin'
})

interface ImportResult {
  title: string
  markdown: string
  url: string
  imageUrls?: string[]
  images?: Array<{ originalUrl: string; localUrl: string }>
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

interface TranslatedBlock {
  original: string
  translated: string
  type: string
  level?: number
  language?: string
}

const url = ref('')
const taskId = ref<number | null>(null)
const status = ref('')
const stage = ref('')
const progress = ref(0)
const preview = ref<ImportResult | null>(null)
const error = ref<string | null>(null)
const proceeding = ref(false)
const isCreating = ref(false)
const isCancelling = ref(false)
const isFetchingImages = ref(false)
const toast = useToast()

// Translation state
const isTranslating = ref(false)
const translateProgress = ref('')
const translatedBlocks = ref<TranslatedBlock[] | null>(null)
const translatedTitle = ref('')
const bilingualMarkdown = ref('')

// Content mode
const contentMode = ref<'original-md' | 'edit' | 'preview-bilingual' | 'bilingual'>('original-md')

const contentModes = [
  { label: 'Original Preview', value: 'original-md' },
  { label: 'Edit Original', value: 'edit' },
  { label: 'Preview Bilingual', value: 'preview-bilingual' },
  { label: 'Edit Bilingual', value: 'bilingual' }
]

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

// Check if there are images
const hasImages = computed(() => {
  return (preview.value?.imageUrls?.length || 0) > 0 || (preview.value?.images?.length || 0) > 0
})

// Check if images are downloaded
const imagesDownloaded = computed(() => {
  return (preview.value?.images?.length || 0) > 0
})

// Configure marked
marked.setOptions({
  breaks: true,
  gfm: true
})

// Render original markdown as HTML
const renderedOriginalMarkdown = computed(() => {
  if (!preview.value?.markdown) return ''
  try {
    const html = marked.parse(preview.value.markdown) as string
    return DOMPurify.sanitize(html)
  } catch {
    return preview.value.markdown
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

  // Reset translation state
  isTranslating.value = false
  translateProgress.value = ''
  translatedBlocks.value = null
  bilingualMarkdown.value = ''
  translatedTitle.value = ''
  contentMode.value = 'original-md'

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

    if (data.status === 'completed' && data.result) {
      preview.value = data.result

      toast.add({
        title: 'Article fetched successfully',
        color: 'success'
      })

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

      if (data.status === 'completed' && data.result) {
        preview.value = data.result

        toast.add({
          title: 'Article fetched successfully',
          color: 'success'
        })
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

const fetchImages = async () => {
  if (!taskId.value || !preview.value) return

  isFetchingImages.value = true

  try {
    await $fetch(`/api/admin/articles/import-queue/${taskId.value}/images`, {
      method: 'POST'
    })

    // Poll for image download completion
    const checkImages = async () => {
      const data = await $fetch(`/api/admin/articles/import-queue/${taskId.value}`)
      if (data.result?.images?.length) {
        preview.value = data.result
        toast.add({
          title: `${data.result.images.length} image(s) downloaded`,
          color: 'success'
        })
      } else if (data.stage === 'images') {
        // Still downloading
        setTimeout(checkImages, 1000)
      }
    }

    setTimeout(checkImages, 1000)
  } catch (e: any) {
    toast.add({
      title: 'Failed to fetch images',
      description: e.data?.message || e.message,
      color: 'error'
    })
  } finally {
    isFetchingImages.value = false
  }
}

const proceedToCreate = async () => {
  if (!preview.value) return

  proceeding.value = true

  try {
    // Determine which content to use
    let contentToUse = preview.value.markdown

    // If bilingual markdown exists and user is in bilingual mode, use that
    if (bilingualMarkdown.value && contentMode.value !== 'original-md' && contentMode.value !== 'edit') {
      contentToUse = bilingualMarkdown.value
    }

    // Store in sessionStorage
    sessionStorage.setItem('importedArticle', JSON.stringify({
      title: preview.value.title,
      content: contentToUse,
      translatedTitle: translatedTitle.value || undefined
    }))

    // Navigate to create page
    await navigateTo('/admin/articles/create')
  } finally {
    proceeding.value = false
  }
}

// Translation function
const startTranslation = async () => {
  if (!preview.value?.markdown) return

  isTranslating.value = true
  translateProgress.value = 'Starting...'
  translatedBlocks.value = null
  bilingualMarkdown.value = ''
  translatedTitle.value = ''

  try {
    const result = await $fetch('/api/admin/articles/translate-markdown', {
      method: 'POST',
      body: {
        markdown: preview.value.markdown,
        title: preview.value.title
      }
    })

    if (result.success) {
      translatedBlocks.value = result.blocks
      bilingualMarkdown.value = result.bilingualMarkdown
      translatedTitle.value = result.translatedTitle
      contentMode.value = 'preview-bilingual'

      toast.add({
        title: 'Translation completed',
        color: 'success'
      })
    }
  } catch (e: any) {
    console.error('Translation error:', e)
    toast.add({
      title: 'Translation failed',
      description: e.data?.message || e.message || 'Unknown error',
      color: 'error'
    })
  } finally {
    isTranslating.value = false
    translateProgress.value = ''
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