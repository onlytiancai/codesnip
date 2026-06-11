<script setup lang="ts">
const route = useRoute()

const { data, error } = await useFetch(`/api/public/articles/${route.params.id}`)

if (error.value || !data.value) {
  throw createError({
    statusCode: 404,
    message: 'Article not found'
  })
}

const article = computed(() => data.value?.article)
</script>

<template>
  <div>
    <div class="mb-6">
      <NuxtLink to="/" class="text-blue-500 hover:text-blue-600 text-sm">
        &larr; Back to Home
      </NuxtLink>
    </div>

    <article class="max-w-4xl mx-auto">
      <header class="mb-8">
        <h1 class="text-3xl font-bold text-gray-900 mb-2">{{ article?.title }}</h1>

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
          <span>By {{ article?.user?.name || 'Anonymous' }}</span>
          <span>{{ new Date(article?.createdAt || '').toLocaleDateString() }}</span>
        </div>

        <p v-if="article?.description" class="mt-4 text-gray-600">
          {{ article.description }}
        </p>
      </header>

      <ThemeSelector />

      <div class="mt-8">
        <MarkdownRenderer :content="article?.content || ''" />
      </div>
    </article>
  </div>
</template>
