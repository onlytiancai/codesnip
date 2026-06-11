<script setup lang="ts">
definePageMeta({
  middleware: ['auth']
})

const { data: articlesData, refresh: refreshArticles } = await useFetch('/api/articles')
const articles = computed(() => articlesData.value?.articles || [])

async function deleteArticle(id: string) {
  if (!confirm('Are you sure you want to delete this article?')) return

  try {
    await $fetch(`/api/articles/${id}`, { method: 'DELETE' })
    await refreshArticles()
  } catch (e) {
    console.error('Failed to delete article:', e)
  }
}

async function togglePublish(id: string) {
  try {
    await $fetch(`/api/articles/${id}/publish`, { method: 'POST' })
    await refreshArticles()
  } catch (e) {
    console.error('Failed to publish article:', e)
  }
}
</script>

<template>
  <div>
    <div class="flex justify-between items-center mb-6">
      <h1 class="text-2xl font-bold text-gray-900">Dashboard</h1>
      <NuxtLink
        to="/articles/new"
        class="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
      >
        Scrape New Article
      </NuxtLink>
    </div>

    <div v-if="articles.length === 0" class="text-center py-12 text-gray-500">
      No articles yet.
      <NuxtLink to="/articles/new" class="text-blue-500 hover:text-blue-600">
        Scrape your first article
      </NuxtLink>
    </div>

    <div class="bg-white shadow rounded-lg overflow-hidden">
      <table v-if="articles.length > 0" class="min-w-full divide-y divide-gray-200">
        <thead class="bg-gray-50">
          <tr>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Title</th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Category</th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-200">
          <tr v-for="article in articles" :key="article.id">
            <td class="px-6 py-4">
              <div class="text-sm font-medium text-gray-900">{{ article.title }}</div>
              <div class="text-sm text-gray-500">{{ article.description?.substring(0, 50) }}...</div>
            </td>
            <td class="px-6 py-4">
              <span
                :class="[
                  'px-2 py-1 text-xs rounded-full',
                  article.isPublished
                    ? 'bg-green-100 text-green-800'
                    : 'bg-gray-100 text-gray-800'
                ]"
              >
                {{ article.isPublished ? 'Published' : 'Draft' }}
              </span>
            </td>
            <td class="px-6 py-4 text-sm text-gray-500">
              {{ article.category?.name || '-' }}
            </td>
            <td class="px-6 py-4 flex gap-2">
              <NuxtLink
                :to="`/articles/${article.id}`"
                class="text-blue-500 hover:text-blue-600 text-sm"
              >
                View
              </NuxtLink>
              <button
                @click="togglePublish(article.id)"
                class="text-green-500 hover:text-green-600 text-sm"
              >
                {{ article.isPublished ? 'Unpublish' : 'Publish' }}
              </button>
              <button
                @click="deleteArticle(article.id)"
                class="text-red-500 hover:text-red-600 text-sm"
              >
                Delete
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
