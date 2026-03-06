<template>
  <NuxtLayout name="admin">
    <div>
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
                <span class="text-sm text-gray-500 dark:text-gray-400">vs last month</span>
              </div>
            </div>
            <div :class="stat.iconBg" class="p-3 rounded-lg">
              <UIcon :name="stat.icon" class="w-6 h-6 text-white" />
            </div>
          </div>
        </UCard>
      </div>

      <!-- Quick Actions -->
      <div class="flex flex-wrap gap-3 mb-8">
        <UButton to="/admin/articles/create" icon="i-lucide-plus">
          New Article
        </UButton>
        <UButton variant="outline" icon="i-lucide-users">
          Manage Users
        </UButton>
        <UButton variant="outline" icon="i-lucide-bar-chart-2">
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
            <template #actions-cell>
              <UButton icon="i-lucide-more-horizontal" color="neutral" variant="ghost" size="xs" />
            </template>
          </UTable>
        </UCard>

        <!-- Quick Stats -->
        <UCard>
          <template #header>
            <h3 class="text-lg font-semibold">Platform Health</h3>
          </template>
          <div class="space-y-4">
            <div v-for="metric in platformMetrics" :key="metric.label">
              <div class="flex items-center justify-between mb-1">
                <span class="text-sm text-gray-500 dark:text-gray-400">{{ metric.label }}</span>
                <span class="text-sm font-medium">{{ metric.value }}%</span>
              </div>
              <UProgress :value="metric.value" :color="metric.color" size="sm" />
            </div>
          </div>
        </UCard>
      </div>

      <!-- Recent Activity -->
      <UCard class="mt-6">
        <template #header>
          <h3 class="text-lg font-semibold">Recent Activity</h3>
        </template>
        <div class="space-y-4">
          <div
            v-for="activity in recentActivity"
            :key="activity.id"
            class="flex items-center gap-4 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition"
          >
            <UAvatar :src="activity.avatar" :alt="activity.user" size="sm" />
            <div class="flex-1">
              <p class="font-medium">{{ activity.user }}</p>
              <p class="text-sm text-gray-500 dark:text-gray-400">{{ activity.action }}</p>
            </div>
            <span class="text-sm text-gray-400">{{ activity.time }}</span>
          </div>
        </div>
      </UCard>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false
})

const stats = [
  {
    title: 'Total Users',
    value: '12,543',
    trend: 12.5,
    icon: 'i-lucide-users',
    iconBg: 'bg-blue-500'
  },
  {
    title: 'Total Articles',
    value: '1,234',
    trend: 8.2,
    icon: 'i-lucide-file-text',
    iconBg: 'bg-green-500'
  },
  {
    title: 'Premium Members',
    value: '892',
    trend: 15.3,
    icon: 'i-lucide-crown',
    iconBg: 'bg-purple-500'
  },
  {
    title: 'Monthly Revenue',
    value: '$8,231',
    trend: -3.1,
    icon: 'i-lucide-dollar-sign',
    iconBg: 'bg-orange-500'
  }
]

const articleColumns = [
  { id: 'title', header: 'Title' },
  { id: 'category', header: 'Category' },
  { id: 'views', header: 'Views' },
  { id: 'status', header: 'Status' },
  { id: 'actions', header: '' }
]

const recentArticles = [
  { id: 1, title: 'AI in Healthcare', category: 'Technology', views: '2.3k', status: 'published' },
  { id: 2, title: 'Climate Change Update', category: 'Science', views: '1.8k', status: 'published' },
  { id: 3, title: 'Startup Success Stories', category: 'Business', views: '3.1k', status: 'draft' },
  { id: 4, title: 'Sleep Science', category: 'Health', views: '4.2k', status: 'published' },
  { id: 5, title: 'Digital Transformation', category: 'Business', views: '1.5k', status: 'draft' }
]

const platformMetrics = [
  { label: 'Server Uptime', value: 99.9, color: 'success' },
  { label: 'API Response Time', value: 85, color: 'success' },
  { label: 'Storage Used', value: 62, color: 'warning' },
  { label: 'Cache Hit Rate', value: 94, color: 'success' }
]

const recentActivity = [
  {
    id: 1,
    user: 'John Doe',
    action: 'Published article "AI in Healthcare"',
    time: '2h ago',
    avatar: 'https://avatars.githubusercontent.com/u/2?v=4'
  },
  {
    id: 2,
    user: 'Jane Smith',
    action: 'Created new category "Environment"',
    time: '3h ago',
    avatar: 'https://avatars.githubusercontent.com/u/3?v=4'
  },
  {
    id: 3,
    user: 'Bob Johnson',
    action: 'Deleted 5 inactive users',
    time: '5h ago',
    avatar: 'https://avatars.githubusercontent.com/u/4?v=4'
  },
  {
    id: 4,
    user: 'Alice Brown',
    action: 'Updated system settings',
    time: '1d ago',
    avatar: 'https://avatars.githubusercontent.com/u/5?v=4'
  }
]
</script>