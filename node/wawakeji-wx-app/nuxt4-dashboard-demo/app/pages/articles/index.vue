<template>
  <NuxtLayout name="default">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Header -->
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 class="text-2xl font-bold">Articles</h1>
          <p class="text-gray-500 dark:text-gray-400">Browse and discover articles for your level</p>
        </div>
        <UInput
          v-model="searchQuery"
          placeholder="Search articles..."
          icon="i-lucide-search"
          class="w-full sm:w-72"
          @keyup.enter="handleSearch"
        />
      </div>

      <!-- Filters -->
      <div class="flex flex-wrap gap-2 mb-6">
        <UButton
          v-for="cat in categoryFilters"
          :key="cat.value"
          :variant="selectedCategory === cat.value ? 'solid' : 'outline'"
          :color="selectedCategory === cat.value ? 'primary' : 'neutral'"
          size="sm"
          @click="selectedCategory = cat.value"
        >
          {{ cat.label }}
        </UButton>
      </div>

      <!-- Difficulty Filter -->
      <div class="flex items-center gap-3 mb-8">
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

      <!-- Loading State -->
      <div v-if="pending" class="flex justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin" />
      </div>

      <!-- Empty State -->
      <div v-else-if="articles.length === 0" class="text-center py-12">
        <UIcon name="i-lucide-file-x" class="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 class="text-lg font-semibold mb-2">No articles found</h3>
        <p class="text-gray-500 dark:text-gray-400">Try adjusting your filters or search query</p>
      </div>

      <!-- Articles Grid -->
      <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
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
                <UBadge color="primary" variant="subtle" size="xs">{{ article.category?.name }}</UBadge>
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
                <div class="flex items-center gap-1">
                  <UIcon name="i-lucide-bookmark" class="w-3 h-3" />
                  <span>{{ article.bookmarks }}</span>
                </div>
              </div>
            </div>
          </UCard>
        </NuxtLink>
      </div>

      <!-- Pagination -->
      <div v-if="totalPages > 1" class="flex justify-center mt-8">
        <UPagination v-model:page="currentPage" :total="total" :items-per-page="limit" />
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const route = useRoute()
const router = useRouter()

// State
const selectedCategory = ref('all')
const selectedDifficulty = ref('all')
const currentPage = ref(1)
const searchQuery = ref('')
const limit = 9

// Initialize search from URL query param
onMounted(() => {
  const q = route.query.q
  if (q && typeof q === 'string') {
    searchQuery.value = q
  }
})

// Category filters
const categoryFilters = ref([
  { label: 'All', value: 'all' }
])

const difficulties = [
  { label: 'All', value: 'all', color: 'neutral' },
  { label: 'Beginner', value: 'beginner', color: 'success' },
  { label: 'Intermediate', value: 'intermediate', color: 'warning' },
  { label: 'Advanced', value: 'advanced', color: 'error' }
]

// Fetch categories for filter buttons
const { data: categoriesData } = await useFetch('/api/categories')
watchEffect(() => {
  if (categoriesData.value) {
    categoryFilters.value = [
      { label: 'All', value: 'all' },
      ...categoriesData.value.map(cat => ({
        label: cat.name,
        value: cat.slug
      }))
    ]
  }
})

// Build query for articles
const articlesQuery = computed(() => {
  const query: Record<string, any> = {
    page: currentPage.value,
    limit
  }
  if (selectedCategory.value !== 'all') {
    query.categorySlug = selectedCategory.value
  }
  if (selectedDifficulty.value !== 'all') {
    query.difficulty = selectedDifficulty.value
  }
  if (searchQuery.value) {
    query.search = searchQuery.value
  }
  return query
})

// Fetch articles
const { data, pending } = await useFetch('/api/articles', {
  query: articlesQuery
})

const articles = computed(() => data.value?.articles || [])
const total = computed(() => data.value?.pagination.total || 0)
const totalPages = computed(() => data.value?.pagination.totalPages || 1)

// Reset page when filters change
watch([selectedCategory, selectedDifficulty], () => {
  currentPage.value = 1
})

const handleSearch = () => {
  currentPage.value = 1
}

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
  title: 'Articles - English Reading',
  description: 'Browse and discover English reading articles for your level'
})
</script>