<template>
  <NuxtLayout name="admin">
    <div>
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
                <p class="text-2xl font-bold mt-1">{{ stats?.totalUsers || 0 }}</p>
                <div class="flex items-center gap-1 mt-2">
                  <UIcon name="i-lucide-trending-up" class="w-4 h-4 text-green-500" />
                  <span class="text-green-500 text-sm">+{{ stats?.newUsersLast30Days || 0 }}</span>
                  <span class="text-sm text-gray-500 dark:text-gray-400">this month</span>
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
                <p class="text-2xl font-bold mt-1">{{ stats?.totalArticles || 0 }}</p>
                <div class="flex items-center gap-1 mt-2">
                  <UIcon name="i-lucide-trending-up" class="w-4 h-4 text-green-500" />
                  <span class="text-green-500 text-sm">+{{ stats?.newArticlesLast30Days || 0 }}</span>
                  <span class="text-sm text-gray-500 dark:text-gray-400">this month</span>
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
                <p class="text-sm text-gray-500 dark:text-gray-400">Published</p>
                <p class="text-2xl font-bold mt-1">{{ stats?.publishedArticles || 0 }}</p>
                <div class="flex items-center gap-1 mt-2">
                  <span class="text-sm text-gray-500 dark:text-gray-400">
                    {{ stats?.totalCategories || 0 }} categories
                  </span>
                </div>
              </div>
              <div class="bg-purple-500 p-3 rounded-lg">
                <UIcon name="i-lucide-check-circle" class="w-6 h-6 text-white" />
              </div>
            </div>
          </UCard>

          <UCard>
            <div class="flex items-center justify-between">
              <div>
                <p class="text-sm text-gray-500 dark:text-gray-400">Total Views</p>
                <p class="text-2xl font-bold mt-1">{{ formatNumber(stats?.totalViews || 0) }}</p>
                <div class="flex items-center gap-1 mt-2">
                  <span class="text-sm text-gray-500 dark:text-gray-400">
                    {{ formatNumber(stats?.totalBookmarks || 0) }} bookmarks
                  </span>
                </div>
              </div>
              <div class="bg-orange-500 p-3 rounded-lg">
                <UIcon name="i-lucide-eye" class="w-6 h-6 text-white" />
              </div>
            </div>
          </UCard>
        </div>

        <!-- Quick Actions -->
        <div class="flex flex-wrap gap-3 mb-8">
          <UButton to="/admin/articles/create" icon="i-lucide-plus">
            New Article
          </UButton>
          <UButton variant="outline" to="/admin/users" icon="i-lucide-users">
            Manage Users
          </UButton>
          <UButton variant="outline" to="/admin/analytics" icon="i-lucide-bar-chart-2">
            View Analytics
          </UButton>
        </div>

        <!-- Main Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <!-- Recent Articles -->
          <UCard class="lg:col-span-2">
            <template #header>
              <div class="flex items-center justify-between">
                <h3 class="text-lg font-semibold">Recent Articles</h3>
                <NuxtLink to="/admin/articles" class="text-sm text-primary hover:underline">
                  View all
                </NuxtLink>
              </div>
            </template>
            <UTable :data="recentArticles" :columns="articleColumns">
              <template #status-cell="{ row }">
                <UBadge :color="row.original.status === 'published' ? 'success' : 'warning'" variant="subtle" size="xs">
                  {{ row.original.status }}
                </UBadge>
              </template>
              <template #category-cell="{ row }">
                <span class="text-sm">{{ row.original.category?.name || '-' }}</span>
              </template>
              <template #actions-cell="{ row }">
                <UButton
                  icon="i-lucide-more-horizontal"
                  color="neutral"
                  variant="ghost"
                  size="xs"
                  :to="`/admin/articles/${row.original.id}/edit`"
                />
              </template>
            </UTable>
          </UCard>

          <!-- Quick Stats -->
          <UCard>
            <template #header>
              <h3 class="text-lg font-semibold">Platform Stats</h3>
            </template>
            <div class="space-y-4">
              <div>
                <div class="flex items-center justify-between mb-1">
                  <span class="text-sm text-gray-500 dark:text-gray-400">Active Categories</span>
                  <span class="text-sm font-medium">{{ stats?.totalCategories || 0 }}</span>
                </div>
                <UProgress :value="stats?.totalCategories || 0" color="success" size="sm" :max="20" />
              </div>
              <div>
                <div class="flex items-center justify-between mb-1">
                  <span class="text-sm text-gray-500 dark:text-gray-400">Total Tags</span>
                  <span class="text-sm font-medium">{{ stats?.totalTags || 0 }}</span>
                </div>
                <UProgress :value="stats?.totalTags || 0" color="primary" size="sm" :max="50" />
              </div>
              <div>
                <div class="flex items-center justify-between mb-1">
                  <span class="text-sm text-gray-500 dark:text-gray-400">Publish Rate</span>
                  <span class="text-sm font-medium">{{ publishRate }}%</span>
                </div>
                <UProgress :value="publishRate" color="warning" size="sm" />
              </div>
            </div>
          </UCard>
        </div>

        <!-- Recent Activity -->
        <UCard class="mt-6">
          <template #header>
            <h3 class="text-lg font-semibold">Recent Users</h3>
          </template>
          <div class="space-y-4">
            <div
              v-for="user in recentUsers"
              :key="user.id"
              class="flex items-center gap-4 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition"
            >
              <UAvatar :src="user.avatar || undefined" :alt="user.name || user.email" size="sm" />
              <div class="flex-1">
                <p class="font-medium">{{ user.name || user.email }}</p>
                <p class="text-sm text-gray-500 dark:text-gray-400">{{ user.email }}</p>
              </div>
              <span class="text-sm text-gray-400">{{ formatDate(user.createdAt) }}</span>
            </div>
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

const { stats, recentArticles, recentUsers, loading, fetchOverview } = useAdminAnalytics()

const articleColumns = [
  { id: 'title', header: 'Title', accessorKey: 'title' },
  { id: 'category', header: 'Category' },
  { id: 'status', header: 'Status' },
  { id: 'actions', header: '' }
]

const publishRate = computed(() => {
  if (!stats.value?.totalArticles) return 0
  return Math.round((stats.value.publishedArticles / stats.value.totalArticles) * 100)
})

const formatNumber = (num: number) => {
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

onMounted(() => {
  fetchOverview()
})
</script>