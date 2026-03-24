<script setup lang="ts">
definePageMeta({
  middleware: ['auth']
})

const url = ref('')
const scrapeImages = ref(true)
const loading = ref(false)
const error = ref('')
const jobId = ref<string | null>(null)
const jobStatus = ref<string | null>(null)
const jobProgress = ref(0)
const articleId = ref<string | null>(null)

async function startScrape() {
  if (!url.value) return

  error.value = ''
  loading.value = true

  try {
    const result = await $fetch<{ job: { id: string } }>('/api/jobs/scrape', {
      method: 'POST',
      body: { url: url.value, scrapeImages: scrapeImages.value }
    })

    jobId.value = result.job.id
    pollJobStatus()
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : 'Failed to start scraping'
    loading.value = false
  }
}

async function pollJobStatus() {
  if (!jobId.value) return

  try {
    const result = await $fetch<{ job: { status: string; progress: number; result?: string } }>(
      `/api/jobs/${jobId.value}`
    )

    jobStatus.value = result.job.status
    jobProgress.value = result.job.progress

    if (result.job.result) {
      const parsed = JSON.parse(result.job.result)
      articleId.value = parsed.articleId
    }

    if (result.job.status === 'completed') {
      loading.value = false
      if (articleId.value) {
        navigateTo(`/articles/${articleId.value}`)
      }
    } else if (result.job.status === 'failed') {
      error.value = 'Scraping failed'
      loading.value = false
    } else {
      setTimeout(pollJobStatus, 1000)
    }
  } catch (e) {
    console.error('Failed to poll job status:', e)
    setTimeout(pollJobStatus, 2000)
  }
}
</script>

<template>
  <div class="max-w-2xl mx-auto">
    <h1 class="text-2xl font-bold text-gray-900 mb-6">Scrape New Article</h1>

    <div class="bg-white shadow rounded-lg p-6">
      <form @submit.prevent="startScrape" class="space-y-4">
        <div v-if="error" class="p-3 bg-red-100 text-red-700 rounded">
          {{ error }}
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">URL to Scrape</label>
          <input
            v-model="url"
            type="url"
            required
            placeholder="https://example.com/article"
            class="w-full px-3 py-2 border border-gray-300 rounded-md"
          />
        </div>

        <div class="flex items-center gap-2">
          <input
            v-model="scrapeImages"
            type="checkbox"
            id="scrapeImages"
            class="w-4 h-4 text-blue-500 border-gray-300 rounded focus:ring-blue-500"
          />
          <label for="scrapeImages" class="text-sm font-medium text-gray-700">
            Download images
          </label>
          <span class="text-xs text-gray-500">(store images locally)</span>
        </div>

        <button
          type="submit"
          :disabled="loading || !url"
          class="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 disabled:opacity-50"
        >
          {{ loading ? 'Scraping...' : 'Start Scraping' }}
        </button>
      </form>

      <JobProgress v-if="loading" :progress="jobProgress" :status="jobStatus || 'processing'" />
    </div>

    <div class="mt-6">
      <p class="text-sm text-gray-500">
        Test URL: <code class="bg-gray-100 px-2 py-1 rounded">https://code.claude.com/docs/en/sub-agents</code>
      </p>
    </div>
  </div>
</template>
