<template>
  <NuxtLayout name="admin">
    <div>
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-2xl font-bold">Analytics</h2>
        <div class="flex items-center gap-3">
          <USelect
            v-model="dateRange"
            :items="dateRangeOptions"
            class="w-40"
            @change="handleDateRangeChange"
          />
          <UButton icon="i-lucide-download" variant="outline">
            Export Report
          </UButton>
        </div>
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="flex items-center justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
      </div>

      <template v-else>
        <!-- Stats Cards -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <UCard>
            <div class="flex items-center justify-between">
              <div>
                <p class="text-sm text-gray-500 dark:text-gray-400">Total Users</p>
                <p class="text-2xl font-bold mt-1">{{ formatNumber(stats?.totalUsers || 0) }}</p>
                <div class="flex items-center gap-1 mt-2">
                  <UIcon name="i-lucide-trending-up" class="w-4 h-4 text-green-500" />
                  <span class="text-green-500 text-sm">+{{ stats?.newUsersLast30Days || 0 }}</span>
                  <span class="text-sm text-gray-500">this month</span>
                </div>
              </div>
              <div class="bg-blue-500 p-3 rounded-lg">
                <UIcon name="i-lucide-users" class="w-6 h-6 text-white" />
              </div>
            </div>
          </UCard>

          <UCard>
            <div class="flex items-center justify-between">
              <div>
                <p class="text-sm text-gray-500 dark:text-gray-400">Total Articles</p>
                <p class="text-2xl font-bold mt-1">{{ formatNumber(stats?.totalArticles || 0) }}</p>
                <div class="flex items-center gap-1 mt-2">
                  <UIcon name="i-lucide-trending-up" class="w-4 h-4 text-green-500" />
                  <span class="text-green-500 text-sm">+{{ stats?.newArticlesLast30Days || 0 }}</span>
                  <span class="text-sm text-gray-500">this month</span>
                </div>
              </div>
              <div class="bg-green-500 p-3 rounded-lg">
                <UIcon name="i-lucide-file-text" class="w-6 h-6 text-white" />
              </div>
            </div>
          </UCard>

          <UCard>
            <div class="flex items-center justify-between">
              <div>
                <p class="text-sm text-gray-500 dark:text-gray-400">Total Views</p>
                <p class="text-2xl font-bold mt-1">{{ formatNumber(stats?.totalViews || 0) }}</p>
              </div>
              <div class="bg-purple-500 p-3 rounded-lg">
                <UIcon name="i-lucide-eye" class="w-6 h-6 text-white" />
              </div>
            </div>
          </UCard>

          <UCard>
            <div class="flex items-center justify-between">
              <div>
                <p class="text-sm text-gray-500 dark:text-gray-400">Total Bookmarks</p>
                <p class="text-2xl font-bold mt-1">{{ formatNumber(stats?.totalBookmarks || 0) }}</p>
              </div>
              <div class="bg-orange-500 p-3 rounded-lg">
                <UIcon name="i-lucide-bookmark" class="w-6 h-6 text-white" />
              </div>
            </div>
          </UCard>
        </div>

        <!-- Charts Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <!-- User Growth Chart -->
          <UCard>
            <template #header>
              <h3 class="font-semibold">User Registrations</h3>
            </template>
            <div class="h-64">
              <div v-if="chartData?.userRegistrations?.length" class="h-full flex items-end gap-1">
                <div
                  v-for="(item, index) in chartData.userRegistrations.slice(-30)"
                  :key="index"
                  class="flex-1 bg-blue-500 rounded-t transition-all hover:bg-blue-600"
                  :style="{ height: `${(item.count / maxRegistrationCount) * 100}%` }"
                  :title="`${item.date}: ${item.count} users`"
                />
              </div>
              <div v-else class="h-full flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="text-center">
                  <UIcon name="i-lucide-bar-chart-2" class="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-2" />
                  <p class="text-gray-500 dark:text-gray-400">No data available</p>
                </div>
              </div>
            </div>
          </UCard>

          <!-- Article Creations Chart -->
          <UCard>
            <template #header>
              <h3 class="font-semibold">Article Creations</h3>
            </template>
            <div class="h-64">
              <div v-if="chartData?.articleCreations?.length" class="h-full flex items-end gap-1">
                <div
                  v-for="(item, index) in chartData.articleCreations.slice(-30)"
                  :key="index"
                  class="flex-1 bg-green-500 rounded-t transition-all hover:bg-green-600"
                  :style="{ height: `${(item.count / maxArticleCount) * 100}%` }"
                  :title="`${item.date}: ${item.count} articles`"
                />
              </div>
              <div v-else class="h-full flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="text-center">
                  <UIcon name="i-lucide-activity" class="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-2" />
                  <p class="text-gray-500 dark:text-gray-400">No data available</p>
                </div>
              </div>
            </div>
          </UCard>
        </div>

        <!-- More Charts -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <!-- Category Distribution -->
          <UCard>
            <template #header>
              <h3 class="font-semibold">Popular Categories</h3>
            </template>
            <div class="space-y-4">
              <div v-for="category in chartData?.categoryDistribution" :key="category.name">
                <div class="flex items-center justify-between mb-1">
                  <span class="text-sm">{{ category.name }}</span>
                  <span class="text-sm font-medium">{{ category.count }}</span>
                </div>
                <UProgress :value="(category.count / maxCategoryCount) * 100" color="primary" size="sm" />
              </div>
              <p v-if="!chartData?.categoryDistribution?.length" class="text-center text-gray-500 py-4">
                No categories yet
              </p>
            </div>
          </UCard>

          <!-- Difficulty Distribution -->
          <UCard>
            <template #header>
              <h3 class="font-semibold">Difficulty Distribution</h3>
            </template>
            <div class="flex items-center justify-center h-48">
              <div v-if="chartData?.difficultyDistribution?.length" class="text-center">
                <div class="space-y-4">
                  <div v-for="item in chartData.difficultyDistribution" :key="item.difficulty" class="flex items-center gap-3">
                    <div
                      class="w-4 h-4 rounded"
                      :class="{
                        'bg-green-500': item.difficulty === 'beginner',
                        'bg-yellow-500': item.difficulty === 'intermediate',
                        'bg-red-500': item.difficulty === 'advanced'
                      }"
                    />
                    <span class="text-sm capitalize">{{ item.difficulty }}</span>
                    <span class="text-sm font-medium">{{ item.count }}</span>
                  </div>
                </div>
              </div>
              <div v-else class="text-center">
                <p class="text-gray-500 dark:text-gray-400">No data available</p>
              </div>
            </div>
          </UCard>

          <!-- Top Articles -->
          <UCard>
            <template #header>
              <h3 class="font-semibold">Top Articles</h3>
            </template>
            <div class="space-y-3">
              <div
                v-for="(article, index) in chartData?.topArticles"
                :key="article.id"
                class="flex items-center gap-3"
              >
                <span class="text-sm font-medium text-gray-400 w-5">{{ index + 1 }}</span>
                <div class="flex-1 min-w-0">
                  <p class="text-sm font-medium truncate">{{ article.title }}</p>
                  <p class="text-xs text-gray-500">{{ formatNumber(article.views) }} views</p>
                </div>
              </div>
              <p v-if="!chartData?.topArticles?.length" class="text-center text-gray-500 py-4">
                No articles yet
              </p>
            </div>
          </UCard>
        </div>

        <!-- Recent Users -->
        <UCard>
          <template #header>
            <h3 class="font-semibold">Recent Registrations</h3>
          </template>
          <div class="space-y-3">
            <div
              v-for="user in recentUsers"
              :key="user.id"
              class="flex items-center gap-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800"
            >
              <UAvatar :src="user.avatar || undefined" :alt="user.name || user.email" size="sm" />
              <div class="flex-1">
                <p class="text-sm font-medium">{{ user.name || user.email }}</p>
                <p class="text-xs text-gray-500">{{ user.email }}</p>
              </div>
              <span class="text-xs text-gray-400">{{ formatDate(user.createdAt) }}</span>
            </div>
            <p v-if="!recentUsers?.length" class="text-center text-gray-500 py-4">
              No users yet
            </p>
          </div>
        </UCard>
      </template>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false,
  middleware: 'admin'
})

const { stats, recentUsers, chartData, loading, fetchOverview, fetchCharts } = useAdminAnalytics()

const dateRange = ref('30')

const dateRangeOptions = [
  { label: 'Last 7 days', value: '7' },
  { label: 'Last 30 days', value: '30' },
  { label: 'Last 90 days', value: '90' }
]

const maxRegistrationCount = computed(() => {
  if (!chartData.value?.userRegistrations?.length) return 1
  return Math.max(...chartData.value.userRegistrations.map(r => r.count), 1)
})

const maxArticleCount = computed(() => {
  if (!chartData.value?.articleCreations?.length) return 1
  return Math.max(...chartData.value.articleCreations.map(r => r.count), 1)
})

const maxCategoryCount = computed(() => {
  if (!chartData.value?.categoryDistribution?.length) return 1
  return Math.max(...chartData.value.categoryDistribution.map(c => c.count), 1)
})

const formatNumber = (num: number) => {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M'
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'k'
  }
  return num.toString()
}

const formatDate = (dateStr: string) => {
  const date = new Date(dateStr)
  const now = new Date()
  const diff = now.getTime() - date.getTime()
  const days = Math.floor(diff / (1000 * 60 * 60 * 24))

  if (days === 0) return 'Today'
  if (days === 1) return 'Yesterday'
  if (days < 7) return `${days} days ago`
  return date.toLocaleDateString()
}

const handleDateRangeChange = () => {
  fetchCharts(parseInt(dateRange.value))
}

onMounted(async () => {
  await Promise.all([
    fetchOverview(),
    fetchCharts(parseInt(dateRange.value))
  ])
})
</script>