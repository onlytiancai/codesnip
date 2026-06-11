<script setup lang="ts">
const { data: articlesData } = await useFetch('/api/public/articles')
const { data: categoriesData } = await useFetch('/api/categories')
const { data: tagsData } = await useFetch('/api/tags')

const selectedCategory = ref<string | null>(null)
const selectedTag = ref<string | null>(null)

const { data: filteredArticles, refresh: refreshArticles } = await useFetch('/api/public/articles', {
  query: {
    category: computed(() => selectedCategory.value),
    tag: computed(() => selectedTag.value)
  },
  watch: [selectedCategory, selectedTag]
})

const articles = computed(() => filteredArticles.value?.articles || [])
const categories = computed(() => categoriesData.value?.categories || [])
const tags = computed(() => tagsData.value?.tags || [])

function selectCategory(slug: string | null) {
  selectedCategory.value = selectedCategory.value === slug ? null : slug
}

function selectTag(slug: string | null) {
  selectedTag.value = selectedTag.value === slug ? null : slug
}
</script>

<template>
  <div>
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-4">Public Articles</h1>

      <div class="flex flex-wrap gap-4 mb-6">
        <div class="flex items-center gap-2">
          <span class="text-sm font-medium text-gray-700">Category:</span>
          <button
            v-for="cat in categories"
            :key="cat.id"
            @click="selectCategory(cat.slug)"
            :class="[
              'px-3 py-1 text-sm rounded-full transition-colors',
              selectedCategory === cat.slug
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            ]"
          >
            {{ cat.name }} ({{ cat.articleCount }})
          </button>
        </div>
      </div>

      <div class="flex flex-wrap gap-2">
        <span class="text-sm font-medium text-gray-700">Tags:</span>
        <button
          v-for="tag in tags"
          :key="tag.id"
          @click="selectTag(tag.slug)"
          :class="[
            'px-3 py-1 text-sm rounded-full transition-colors',
            selectedTag === tag.slug
              ? 'bg-green-500 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          ]"
        >
          {{ tag.name }} ({{ tag.articleCount }})
        </button>
      </div>
    </div>

    <div v-if="articles.length === 0" class="text-center py-12 text-gray-500">
      No articles found.
    </div>

    <div class="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      <ArticleCard
        v-for="article in articles"
        :key="article.id"
        :article="article"
      />
    </div>
  </div>
</template>
