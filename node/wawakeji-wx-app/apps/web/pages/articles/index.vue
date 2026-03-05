<template>
  <div class="min-h-screen bg-gray-50">
    <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Header -->
      <div class="mb-8">
        <h1 class="text-3xl font-bold text-gray-900 mb-2">文章列表</h1>
        <p class="text-gray-600">精选技术文章，提升英语阅读能力</p>
      </div>

      <!-- Filters -->
      <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-4 mb-6">
        <div class="flex flex-wrap gap-4">
          <!-- Category Filter -->
          <div class="flex-1 min-w-[200px]">
            <label class="block text-sm font-medium text-gray-700 mb-2">分类</label>
            <select
              v-model="selectedCategory"
              class="w-full rounded-lg border-gray-300 border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">全部分类</option>
              <option v-for="cat in ARTICLE_CATEGORIES" :key="cat.id" :value="cat.id">
                {{ cat.name }}
              </option>
            </select>
          </div>

          <!-- Difficulty Filter -->
          <div class="flex-1 min-w-[200px]">
            <label class="block text-sm font-medium text-gray-700 mb-2">难度</label>
            <select
              v-model="selectedDifficulty"
              class="w-full rounded-lg border-gray-300 border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">全部难度</option>
              <option value="beginner">初级</option>
              <option value="intermediate">中级</option>
              <option value="advanced">高级</option>
            </select>
          </div>

          <!-- Search -->
          <div class="flex-1 min-w-[200px]">
            <label class="block text-sm font-medium text-gray-700 mb-2">搜索</label>
            <input
              v-model="searchQuery"
              type="text"
              placeholder="搜索文章..."
              class="w-full rounded-lg border-gray-300 border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              @keyup.enter="handleSearch"
            />
          </div>

          <!-- Search Button -->
          <div class="flex items-end">
            <button
              @click="handleSearch"
              class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              搜索
            </button>
          </div>
        </div>
      </div>

      <!-- Articles Grid -->
      <div v-if="loading" class="text-center py-12">
        <div class="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-600 border-t-transparent"></div>
        <p class="text-gray-600 mt-4">加载中...</p>
      </div>

      <div v-else-if="articles.length === 0" class="text-center py-12">
        <div class="text-6xl mb-4">📭</div>
        <p class="text-gray-600">暂无文章</p>
      </div>

      <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <NuxtLink
          v-for="article in articles"
          :key="article.id"
          :to="`/articles/${article.slug}`"
          class="bg-white border border-gray-200 rounded-xl overflow-hidden hover:shadow-lg transition-all group"
        >
          <div v-if="article.coverImage" class="aspect-video bg-gray-100 overflow-hidden">
            <img
              :src="article.coverImage"
              :alt="article.title"
              class="w-full h-full object-cover group-hover:scale-105 transition-transform"
            />
          </div>
          <div v-else class="aspect-video bg-gradient-to-br from-blue-100 to-purple-100 flex items-center justify-center">
            <span class="text-4xl">📝</span>
          </div>
          <div class="p-4">
            <div class="flex items-center space-x-2 mb-2">
              <span class="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full">
                {{ getCategoryName(article.category) }}
              </span>
              <span
                class="px-2 py-1 text-xs rounded-full"
                :class="{
                  'bg-green-100 text-green-700': article.difficulty === 'beginner',
                  'bg-yellow-100 text-yellow-700': article.difficulty === 'intermediate',
                  'bg-red-100 text-red-700': article.difficulty === 'advanced',
                }"
              >
                {{ getDifficultyLabel(article.difficulty) }}
              </span>
            </div>
            <h3 class="font-semibold text-gray-900 mb-2 line-clamp-2 group-hover:text-blue-600 transition-colors">
              {{ article.title }}
            </h3>
            <p class="text-gray-600 text-sm line-clamp-2 mb-3">
              {{ article.summary }}
            </p>
            <div class="text-xs text-gray-500">
              {{ formatDate(article.publishedAt) }}
            </div>
          </div>
        </NuxtLink>
      </div>

      <!-- Pagination -->
      <div v-if="totalPages > 1" class="flex justify-center items-center space-x-2 mt-8">
        <button
          :disabled="currentPage === 1"
          @click="goToPage(currentPage - 1)"
          class="px-4 py-2 rounded-lg border border-gray-300 text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
        >
          上一页
        </button>

        <span class="text-sm text-gray-600">
          第 {{ currentPage }} / {{ totalPages }} 页
        </span>

        <button
          :disabled="currentPage === totalPages"
          @click="goToPage(currentPage + 1)"
          class="px-4 py-2 rounded-lg border border-gray-300 text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
        >
          下一页
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ARTICLE_CATEGORIES, formatDate, getDifficultyLabel } from '@wawakeji/shared'

definePageMeta({
  layout: 'default',
})

const route = useRoute()
const router = useRouter()

// Query params
const selectedCategory = ref<string>(route.query.category as string || '')
const selectedDifficulty = ref<string>(route.query.difficulty as string || '')
const searchQuery = ref<string>(route.query.search as string || '')
const currentPage = ref<number>(Number(route.query.page) || 1)

const pageSize = 9

// Fetch articles
const { data: articlesData, pending: loading, refresh } = await useFetch('/api/articles', {
  query: computed(() => ({
    category: selectedCategory.value || undefined,
    difficulty: selectedDifficulty.value || undefined,
    search: searchQuery.value || undefined,
    page: currentPage.value,
    pageSize,
  })),
  default: () => ({ items: [], total: 0 }),
})

const articles = computed(() => articlesData.value.items)
const total = computed(() => articlesData.value.total)
const totalPages = computed(() => Math.ceil(total.value / pageSize))

const getCategoryName = (categoryId: string) => {
  const category = ARTICLE_CATEGORIES.find((c) => c.id === categoryId)
  return category?.name || categoryId
}

// Search handler
const handleSearch = () => {
  currentPage.value = 1
  updateQuery()
  refresh()
}

// Page navigation
const goToPage = (page: number) => {
  if (page < 1 || page > totalPages.value) return
  currentPage.value = page
  updateQuery()
  refresh()
}

// Update query params
const updateQuery = () => {
  const query: Record<string, string> = {}
  if (selectedCategory.value) query.category = selectedCategory.value
  if (selectedDifficulty.value) query.difficulty = selectedDifficulty.value
  if (searchQuery.value) query.search = searchQuery.value
  if (currentPage.value > 1) query.page = currentPage.value.toString()

  router.push({ path: '/articles', query })
}

// Watch for query changes
watch(
  () => route.query,
  (newQuery) => {
    if (newQuery.category !== selectedCategory.value) {
      selectedCategory.value = newQuery.category as string
    }
    if (newQuery.difficulty !== selectedDifficulty.value) {
      selectedDifficulty.value = newQuery.difficulty as string
    }
    if (newQuery.search !== searchQuery.value) {
      searchQuery.value = newQuery.search as string
    }
    if (newQuery.page) {
      currentPage.value = Number(newQuery.page)
    }
    refresh()
  },
  { deep: true }
)
</script>
