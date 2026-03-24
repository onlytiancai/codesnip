<script setup lang="ts">
interface Article {
  id: string
  title: string
  description?: string
  isPublished: boolean
  category?: {
    name: string
    slug: string
  }
  tags?: Array<{
    name: string
    slug: string
  }>
  user?: {
    name?: string
  }
  createdAt: string
}

defineProps<{
  article: Article
}>()
</script>

<template>
  <NuxtLink
    :to="`/public/articles/${article.id}`"
    class="block bg-white rounded-lg shadow hover:shadow-md transition-shadow p-6"
  >
    <h3 class="text-lg font-semibold text-gray-900 mb-2 line-clamp-2">
      {{ article.title }}
    </h3>

    <p v-if="article.description" class="text-sm text-gray-600 mb-4 line-clamp-3">
      {{ article.description }}
    </p>

    <div class="flex items-center gap-2 text-sm text-gray-500">
      <span v-if="article.category" class="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs">
        {{ article.category.name }}
      </span>
      <span v-for="tag in article.tags?.slice(0, 3)" :key="tag.slug" class="px-2 py-1 bg-gray-100 rounded text-xs">
        {{ tag.name }}
      </span>
    </div>

    <div class="mt-4 flex items-center justify-between text-xs text-gray-500">
      <span>{{ article.user?.name || 'Anonymous' }}</span>
      <span>{{ new Date(article.createdAt).toLocaleDateString() }}</span>
    </div>
  </NuxtLink>
</template>
