<template>
  <NuxtLayout name="admin">
    <div>
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-2xl font-bold">Analytics</h2>
        <div class="flex items-center gap-3">
          <USelect
            :items="dateRangeOptions"
            default-value="last30"
            class="w-40"
          />
          <UButton icon="i-lucide-download" variant="outline">
            Export Report
          </UButton>
        </div>
      </div>

      <!-- Stats Cards -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <UCard v-for="stat in stats" :key="stat.title">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm text-gray-500 dark:text-gray-400">{{ stat.title }}</p>
              <p class="text-2xl font-bold mt-1">{{ stat.value }}</p>
              <div class="flex items-center gap-1 mt-2">
                <UIcon
                  :name="stat.trend > 0 ? 'i-lucide-trending-up' : 'i-lucide-trending-down'"
                  :class="stat.trend > 0 ? 'text-green-500' : 'text-red-500'"
                  class="w-4 h-4"
                />
                <span :class="stat.trend > 0 ? 'text-green-500' : 'text-red-500'" class="text-sm">
                  {{ Math.abs(stat.trend) }}%
                </span>
              </div>
            </div>
            <div :class="stat.iconBg" class="p-3 rounded-lg">
              <UIcon :name="stat.icon" class="w-6 h-6 text-white" />
            </div>
          </div>
        </UCard>
      </div>

      <!-- Charts Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <!-- User Growth Chart -->
        <UCard>
          <template #header>
            <h3 class="font-semibold">User Growth</h3>
          </template>
          <div class="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div class="text-center">
              <UIcon name="i-lucide-bar-chart-2" class="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-2" />
              <p class="text-gray-500 dark:text-gray-400">Chart Placeholder</p>
              <p class="text-xs text-gray-400">Integrate with Chart.js or similar</p>
            </div>
          </div>
        </UCard>

        <!-- Reading Activity Chart -->
        <UCard>
          <template #header>
            <h3 class="font-semibold">Reading Activity</h3>
          </template>
          <div class="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div class="text-center">
              <UIcon name="i-lucide-activity" class="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-2" />
              <p class="text-gray-500 dark:text-gray-400">Chart Placeholder</p>
              <p class="text-xs text-gray-400">Integrate with Chart.js or similar</p>
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
            <div v-for="category in categoryStats" :key="category.name">
              <div class="flex items-center justify-between mb-1">
                <span class="text-sm">{{ category.name }}</span>
                <span class="text-sm font-medium">{{ category.count }}</span>
              </div>
              <UProgress :value="category.percentage" color="primary" size="sm" />
            </div>
          </div>
        </UCard>

        <!-- Difficulty Distribution -->
        <UCard>
          <template #header>
            <h3 class="font-semibold">Difficulty Distribution</h3>
          </template>
          <div class="flex items-center justify-center h-48">
            <div class="text-center">
              <div class="relative w-32 h-32 mx-auto">
                <!-- Donut chart placeholder -->
                <svg viewBox="0 0 36 36" class="w-32 h-32">
                  <circle cx="18" cy="18" r="15.9" fill="none" stroke="#22c55e" stroke-width="3" stroke-dasharray="40 60" />
                  <circle cx="18" cy="18" r="15.9" fill="none" stroke="#f59e0b" stroke-width="3" stroke-dasharray="35 65" stroke-dashoffset="-40" />
                  <circle cx="18" cy="18" r="15.9" fill="none" stroke="#ef4444" stroke-width="3" stroke-dasharray="25 75" stroke-dashoffset="-75" />
                </svg>
              </div>
              <div class="flex justify-center gap-4 mt-4 text-xs">
                <div class="flex items-center gap-1">
                  <div class="w-3 h-3 rounded-full bg-green-500" />
                  <span>Beginner 40%</span>
                </div>
                <div class="flex items-center gap-1">
                  <div class="w-3 h-3 rounded-full bg-yellow-500" />
                  <span>Intermediate 35%</span>
                </div>
                <div class="flex items-center gap-1">
                  <div class="w-3 h-3 rounded-full bg-red-500" />
                  <span>Advanced 25%</span>
                </div>
              </div>
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
              v-for="(article, index) in topArticles"
              :key="article.id"
              class="flex items-center gap-3"
            >
              <span class="text-sm font-medium text-gray-400 w-5">{{ index + 1 }}</span>
              <div class="flex-1 min-w-0">
                <p class="text-sm font-medium truncate">{{ article.title }}</p>
                <p class="text-xs text-gray-500">{{ article.views }} views</p>
              </div>
            </div>
          </div>
        </UCard>
      </div>

      <!-- Data Tables -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Recent Registrations -->
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
              <UAvatar :src="user.avatar" :alt="user.name" size="sm" />
              <div class="flex-1">
                <p class="text-sm font-medium">{{ user.name }}</p>
                <p class="text-xs text-gray-500">{{ user.email }}</p>
              </div>
              <span class="text-xs text-gray-400">{{ user.joined }}</span>
            </div>
          </div>
        </UCard>

        <!-- Revenue Summary -->
        <UCard>
          <template #header>
            <h3 class="font-semibold">Revenue Summary</h3>
          </template>
          <div class="space-y-4">
            <div class="grid grid-cols-2 gap-4">
              <div class="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <p class="text-2xl font-bold">$8,231</p>
                <p class="text-sm text-gray-500">This Month</p>
              </div>
              <div class="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <p class="text-2xl font-bold">$72,450</p>
                <p class="text-sm text-gray-500">This Year</p>
              </div>
            </div>
            <div class="border-t border-gray-200 dark:border-gray-700 pt-4">
              <h4 class="text-sm font-medium mb-3">By Plan</h4>
              <div class="space-y-2">
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-500">Pro Monthly</span>
                  <span class="text-sm font-medium">$4,860</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-500">Pro Annual</span>
                  <span class="text-sm font-medium">$3,371</span>
                </div>
              </div>
            </div>
          </div>
        </UCard>
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false
})

const dateRangeOptions = [
  { label: 'Last 7 days', value: 'last7' },
  { label: 'Last 30 days', value: 'last30' },
  { label: 'Last 90 days', value: 'last90' },
  { label: 'This Year', value: 'year' }
]

const stats = [
  { title: 'Page Views', value: '54,234', trend: 12.5, icon: 'i-lucide-eye', iconBg: 'bg-blue-500' },
  { title: 'Unique Visitors', value: '12,345', trend: 8.2, icon: 'i-lucide-users', iconBg: 'bg-green-500' },
  { title: 'Avg. Session Duration', value: '8m 32s', trend: -3.1, icon: 'i-lucide-clock', iconBg: 'bg-purple-500' },
  { title: 'Bounce Rate', value: '32%', trend: -5.4, icon: 'i-lucide-arrow-down-right', iconBg: 'bg-orange-500' }
]

const categoryStats = [
  { name: 'Technology', count: '12,450', percentage: 85 },
  { name: 'Business', count: '8,230', percentage: 65 },
  { name: 'Science', count: '6,120', percentage: 48 },
  { name: 'Health', count: '5,890', percentage: 42 },
  { name: 'Culture', count: '3,450', percentage: 28 }
]

const topArticles = [
  { id: 1, title: 'AI in Healthcare', views: '12,450' },
  { id: 2, title: 'Climate Change Facts', views: '8,230' },
  { id: 3, title: 'Startup Success Stories', views: '6,120' },
  { id: 4, title: 'Sleep Science', views: '5,890' },
  { id: 5, title: 'Digital Transformation', views: '3,450' }
]

const recentUsers = [
  { id: 1, name: 'Emily Chen', email: 'emily@example.com', avatar: 'https://avatars.githubusercontent.com/u/6?v=4', joined: '2h ago' },
  { id: 2, name: 'David Lee', email: 'david@example.com', avatar: 'https://avatars.githubusercontent.com/u/7?v=4', joined: '5h ago' },
  { id: 3, name: 'Sarah Kim', email: 'sarah@example.com', avatar: 'https://avatars.githubusercontent.com/u/8?v=4', joined: '1d ago' },
  { id: 4, name: 'Michael Wang', email: 'michael@example.com', avatar: 'https://avatars.githubusercontent.com/u/9?v=4', joined: '2d ago' }
]
</script>