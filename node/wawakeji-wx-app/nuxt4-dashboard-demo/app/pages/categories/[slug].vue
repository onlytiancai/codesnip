<template>
  <NuxtLayout name="default">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Loading State -->
      <div v-if="pendingCategory" class="flex justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin" />
      </div>

      <!-- Error State -->
      <div v-else-if="categoryError || !categoryData" class="text-center py-12">
        <UIcon name="i-lucide-folder-x" class="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 class="text-lg font-semibold mb-2">Category not found</h3>
        <p class="text-gray-500 dark:text-gray-400 mb-4">The category you're looking for doesn't exist.</p>
        <UButton to="/articles">Browse All Articles</UButton>
      </div>

      <template v-else>
        <!-- Category Header -->
        <div class="flex items-center gap-4 mb-8">
          <div :style="{ backgroundColor: category.color }" class="w-16 h-16 rounded-xl flex items-center justify-center opacity-80">
            <UIcon :name="category.icon || 'i-lucide-folder'" class="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 class="text-3xl font-bold">{{ category.name }}</h1>
            <p class="text-gray-500 dark:text-gray-400">{{ category.description }}</p>
          </div>
        </div>

        <!-- Filters -->
        <div class="flex flex-wrap items-center gap-4 mb-8">
          <div class="flex items-center gap-2">
            <span class="text-sm text-gray-500 dark:text-gray-400">Sort by:</span>
            <USelect
              v-model="sortBy"
              :items="[
                { label: 'Newest', value: 'newest' },
                { label: 'Most Popular', value: 'popular' },
                { label: 'Most Saved', value: 'saved' }
              ]"
              size="sm"
              class="w-36"
            />
          </div>
          <div class="flex items-center gap-2">
            <span class="text-sm text-gray-500 dark:text-gray-400">Difficulty:</span>
            <UBadge
              v-for="diff in difficulties"
              :key="diff.value"
              :color="diff.color"
              :variant="selectedDifficulty === diff.value ? 'solid' : 'subtle'"
              class="cursor-pointer"
              @click="selectedDifficulty = diff.value"
            >
              {{ diff.label }}
            </UBadge>
          </div>
        </div>

        <!-- Loading Articles -->
        <div v-if="pendingArticles" class="flex justify-center py-12">
          <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin" />
        </div>

        <!-- Empty State -->
        <div v-else-if="articles.length === 0" class="text-center py-12">
          <UIcon name="i-lucide-file-x" class="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 class="text-lg font-semibold mb-2">No articles in this category</h3>
          <p class="text-gray-500 dark:text-gray-400">Check back later for new content</p>
        </div>

        <!-- Articles Grid -->
        <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <NuxtLink
            v-for="article in articles"
            :key="article.id"
            :to="`/articles/${article.slug}`"
          >
            <UCard class="group h-full hover:border-primary transition cursor-pointer overflow-hidden">
              <img
                :src="article.cover"
                :alt="article.title"
                class="w-full h-48 object-cover group-hover:scale-105 transition duration-300"
              />
              <div class="p-4">
                <div class="flex items-center gap-2 mb-3">
                  <UBadge :color="difficultyColor(article.difficulty)" variant="subtle" size="xs">
                    {{ capitalize(article.difficulty) }}
                  </UBadge>
                </div>
                <h3 class="font-semibold mb-2 line-clamp-2 group-hover:text-primary transition">
                  {{ article.title }}
                </h3>
                <p class="text-sm text-gray-500 dark:text-gray-400 line-clamp-2 mb-4">
                  {{ article.excerpt }}
                </p>
                <div class="flex items-center justify-between text-xs text-gray-400">
                  <div class="flex items-center gap-1">
                    <UIcon name="i-lucide-clock" class="w-3 h-3" />
                    <span>{{ article.readTime }} min</span>
                  </div>
                  <div class="flex items-center gap-1">
                    <UIcon name="i-lucide-eye" class="w-3 h-3" />
                    <span>{{ article.views }}</span>
                  </div>
                </div>
              </div>
            </UCard>
          </NuxtLink>
        </div>

        <!-- Pagination -->
        <div v-if="totalPages > 1" class="flex justify-center">
          <UPagination v-model:page="currentPage" :total="total" :items-per-page="limit" />
        </div>
      </template>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const route = useRoute()

const selectedDifficulty = ref('all')
const currentPage = ref(1)
const sortBy = ref('newest')
const limit = 9

const difficulties = [
  { label: 'All', value: 'all', color: 'neutral' },
  { label: 'Beginner', value: 'beginner', color: 'success' },
  { label: 'Intermediate', value: 'intermediate', color: 'warning' },
  { label: 'Advanced', value: 'advanced', color: 'error' }
]

const slug = computed(() => route.params.slug as string)

// Fetch category
const { data: categoryData, pending: pendingCategory, error: categoryError } = await useFetch(`/api/categories/${slug.value}`)

const category = computed(() => categoryData.value?.category || null)

// Build query for articles
const articlesQuery = computed(() => {
  const query: Record<string, any> = {
    page: currentPage.value,
    limit
  }
  if (selectedDifficulty.value !== 'all') {
    query.difficulty = selectedDifficulty.value
  }
  return query
})

// Fetch articles for this category
const { data: articlesData, pending: pendingArticles } = await useFetch(`/api/categories/${slug.value}`, {
  query: articlesQuery
})

const articles = computed(() => articlesData.value?.articles || [])
const total = computed(() => articlesData.value?.pagination?.total || 0)
const totalPages = computed(() => articlesData.value?.pagination?.totalPages || 1)

// Reset page when filters change
watch(selectedDifficulty, () => {
  currentPage.value = 1
})

const difficultyColor = (difficulty: string) => {
  switch (difficulty) {
    case 'beginner': return 'success'
    case 'intermediate': return 'warning'
    case 'advanced': return 'error'
    default: return 'neutral'
  }
}

const capitalize = (str: string) => {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : ''
}

// SEO
useSeoMeta({
  title: () => category.value ? `${category.value.name} - English Reading` : 'Category',
  description: () => category.value?.description || ''
})
</script>