<template>
  <NuxtLayout name="default">
    <div>
      <!-- Hero Section -->
      <section class="bg-gradient-to-br from-primary/10 to-primary/5 py-16 sm:py-24">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div class="text-center">
            <h1 class="text-4xl sm:text-5xl font-bold mb-6">
              Master English Through
              <span class="text-primary">Reading</span>
            </h1>
            <p class="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto mb-8">
              Improve your English reading skills with curated articles, interactive vocabulary tools, and personalized progress tracking.
            </p>
            <div class="flex flex-col sm:flex-row gap-4 justify-center">
              <UButton size="lg" to="/articles">
                Start Reading
                <UIcon name="i-lucide-arrow-right" class="ml-2 w-4 h-4" />
              </UButton>
              <UButton size="lg" variant="outline" to="/membership">
                Go Premium
              </UButton>
            </div>
            <!-- Search Bar -->
            <div class="mt-8 max-w-xl mx-auto">
              <form @submit.prevent="handleSearch">
                <UInput
                  v-model="homeSearchQuery"
                  placeholder="Search for articles..."
                  icon="i-lucide-search"
                  size="lg"
                  class="w-full"
                >
                  <template #trailing>
                    <UButton type="submit" color="primary" size="sm">
                      Search
                    </UButton>
                  </template>
                </UInput>
              </form>
            </div>
          </div>
        </div>
      </section>

      <!-- Categories Section -->
      <section class="py-16">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 class="text-2xl font-bold mb-8">Browse by Category</h2>
          <div v-if="pendingCategories" class="flex justify-center">
            <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin" />
          </div>
          <div v-else class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
            <NuxtLink
              v-for="category in categories"
              :key="category.slug"
              :to="`/categories/${category.slug}`"
              class="group"
            >
              <UCard class="text-center hover:border-primary transition cursor-pointer">
                <div :style="{ backgroundColor: category.color }" class="w-12 h-12 rounded-lg mx-auto mb-3 flex items-center justify-center opacity-80">
                  <UIcon :name="category.icon || 'i-lucide-folder'" class="w-6 h-6 text-white" />
                </div>
                <h3 class="font-medium group-hover:text-primary transition">{{ category.name }}</h3>
                <p class="text-sm text-gray-500 dark:text-gray-400">{{ category.articleCount }} articles</p>
              </UCard>
            </NuxtLink>
          </div>
        </div>
      </section>

      <!-- Recommended Articles -->
      <section class="py-16 bg-gray-100 dark:bg-gray-900">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div class="flex items-center justify-between mb-8">
            <h2 class="text-2xl font-bold">Recommended for You</h2>
            <NuxtLink to="/articles" class="text-primary hover:underline text-sm font-medium">
              View all
            </NuxtLink>
          </div>
          <div v-if="pendingArticles" class="flex justify-center">
            <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin" />
          </div>
          <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <NuxtLink
              v-for="article in articles.slice(0, 6)"
              :key="article.id"
              :to="`/articles/${article.slug}`"
            >
              <UCard class="group overflow-hidden h-full">
                <img :src="article.cover" :alt="article.title" class="w-full h-48 object-cover group-hover:scale-105 transition duration-300" />
                <template #footer>
                  <div class="p-4">
                    <div class="flex items-center gap-2 mb-2">
                      <UBadge color="primary" variant="subtle" size="xs">{{ article.category?.name }}</UBadge>
                      <UBadge :color="difficultyColor(article.difficulty)" variant="subtle" size="xs">
                        {{ capitalize(article.difficulty) }}
                      </UBadge>
                    </div>
                    <h3 class="font-semibold mb-2 line-clamp-2 group-hover:text-primary transition">
                      {{ article.title }}
                    </h3>
                    <p class="text-sm text-gray-500 dark:text-gray-400 line-clamp-2 mb-3">
                      {{ article.excerpt }}
                    </p>
                    <div class="flex items-center justify-between text-sm text-gray-400">
                      <span>{{ article.readTime }} min read</span>
                      <span>{{ article.views }} views</span>
                    </div>
                  </div>
                </template>
              </UCard>
            </NuxtLink>
          </div>
        </div>
      </section>

      <!-- Features Section -->
      <section class="py-16">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 class="text-2xl font-bold text-center mb-12">Why Choose Us?</h2>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div class="text-center">
              <div class="w-16 h-16 bg-blue-100 dark:bg-blue-900 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                <UIcon name="i-lucide-headphones" class="w-8 h-8 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 class="text-lg font-semibold mb-2">Audio Narration</h3>
              <p class="text-gray-600 dark:text-gray-400">Listen to native speakers while reading to improve pronunciation.</p>
            </div>
            <div class="text-center">
              <div class="w-16 h-16 bg-green-100 dark:bg-green-900 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                <UIcon name="i-lucide-book-marked" class="w-8 h-8 text-green-600 dark:text-green-400" />
              </div>
              <h3 class="text-lg font-semibold mb-2">Vocabulary Builder</h3>
              <p class="text-gray-600 dark:text-gray-400">Save and review new words with spaced repetition.</p>
            </div>
            <div class="text-center">
              <div class="w-16 h-16 bg-purple-100 dark:bg-purple-900 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                <UIcon name="i-lucide-trending-up" class="w-8 h-8 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 class="text-lg font-semibold mb-2">Progress Tracking</h3>
              <p class="text-gray-600 dark:text-gray-400">Track your reading time, words learned, and streaks.</p>
            </div>
          </div>
        </div>
      </section>

      <!-- CTA Section -->
      <section class="py-16 bg-primary text-white">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 class="text-3xl font-bold mb-4">Ready to Improve Your English?</h2>
          <p class="text-lg opacity-90 mb-8 max-w-2xl mx-auto">
            Join thousands of learners who have improved their reading skills with our platform.
          </p>
          <UButton size="lg" variant="solid" color="white" to="/register">
            Get Started Free
          </UButton>
        </div>
      </section>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const router = useRouter()

const homeSearchQuery = ref('')

// Fetch categories
const { data: categoriesData, pending: pendingCategories } = await useFetch('/api/categories')
const categories = computed(() => categoriesData.value || [])

// Fetch recommended articles
const { data: articlesData, pending: pendingArticles } = await useFetch('/api/articles', {
  query: { limit: 6 }
})
const articles = computed(() => articlesData.value?.articles || [])

const handleSearch = () => {
  if (homeSearchQuery.value.trim()) {
    router.push(`/articles?q=${encodeURIComponent(homeSearchQuery.value.trim())}`)
  }
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
  return str.charAt(0).toUpperCase() + str.slice(1)
}
</script>