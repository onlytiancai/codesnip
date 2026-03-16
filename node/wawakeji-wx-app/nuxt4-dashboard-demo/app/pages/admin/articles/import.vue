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
            :disabled="loading"
            @keyup.enter="fetchArticle"
          />
          <UButton
            color="primary"
            :loading="loading"
            :disabled="!url || loading"
            @click="fetchArticle"
          >
            Fetch
          </UButton>
        </div>
      </UCard>

      <!-- Loading State -->
      <div v-if="loading" class="flex items-center justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
        <span class="ml-3 text-gray-500">Fetching and processing article...</span>
      </div>

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

const url = ref('')
const loading = ref(false)
const proceeding = ref(false)
const preview = ref<ImportResult | null>(null)
const error = ref<string | null>(null)
const toast = useToast()

const fetchArticle = async () => {
  if (!url.value) return

  // Reset state
  error.value = null
  preview.value = null

  // Validate URL
  try {
    new URL(url.value)
  } catch {
    error.value = 'Please enter a valid URL'
    return
  }

  loading.value = true

  try {
    const result = await $fetch('/api/admin/articles/import', {
      method: 'POST',
      body: { url: url.value }
    })

    preview.value = result as ImportResult

    if (result.images?.length > 0) {
      toast.add({
        title: `${result.images.length} image(s) downloaded`,
        color: 'success'
      })
    }
  } catch (e: any) {
    error.value = e.data?.message || e.message || 'Failed to fetch article'
    toast.add({
      title: 'Failed to import article',
      description: error.value,
      color: 'error'
    })
  } finally {
    loading.value = false
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
</script>