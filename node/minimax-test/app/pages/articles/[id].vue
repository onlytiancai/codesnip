<script setup lang="ts">
const route = useRoute()
const { loggedIn, user } = useUserSession()

const { data, error, refresh } = await useFetch(`/api/articles/${route.params.id}`)

if (error.value || !data.value) {
  throw createError({
    statusCode: 404,
    message: 'Article not found'
  })
}

const article = computed(() => data.value?.article)
const isOwner = computed(() => loggedIn.value && user.value?.id === article.value?.userId)
const canView = computed(() => article.value?.isPublished || isOwner.value)

if (!canView.value) {
  throw createError({
    statusCode: 403,
    message: 'Access denied'
  })
}

async function togglePublish() {
  try {
    await $fetch(`/api/articles/${article.value?.id}/publish`, { method: 'POST' })
    await refresh()
  } catch (e) {
    console.error('Failed to toggle publish:', e)
  }
}
</script>

<template>
  <div>
    <div class="mb-6">
      <NuxtLink to="/dashboard" class="text-blue-500 hover:text-blue-600 text-sm">
        &larr; Back to Dashboard
      </NuxtLink>
    </div>

    <article class="max-w-4xl mx-auto">
      <header class="mb-8">
        <div class="flex justify-between items-start">
          <h1 class="text-3xl font-bold text-gray-900 mb-2">{{ article?.title }}</h1>
          <div v-if="isOwner" class="flex gap-2">
            <button
              @click="togglePublish"
              :class="[
                'px-4 py-2 rounded-md text-sm',
                article?.isPublished
                  ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  : 'bg-green-500 text-white hover:bg-green-600'
              ]"
            >
              {{ article?.isPublished ? 'Unpublish' : 'Publish' }}
            </button>
          </div>
        </div>

        <div class="flex items-center gap-4 text-sm text-gray-500">
          <span v-if="article?.category">{{ article.category.name }}</span>
          <div v-if="article?.tags?.length" class="flex gap-2">
            <span
              v-for="tag in article.tags"
              :key="tag.id"
              class="px-2 py-1 bg-gray-100 rounded-full text-xs"
            >
              {{ tag.name }}
            </span>
          </div>
          <span>{{ new Date(article?.createdAt || '').toLocaleDateString() }}</span>
        </div>

        <p v-if="article?.description" class="mt-4 text-gray-600">
          {{ article.description }}
        </p>
      </header>

      <div v-if="!article?.isPublished && isOwner" class="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <p class="text-yellow-800 text-sm">
          This article is not published. Only you can see it.
        </p>
      </div>

      <ThemeSelector />

      <div class="mt-8">
        <MarkdownRenderer :content="article?.content || ''" />
      </div>
    </article>
  </div>
</template>
