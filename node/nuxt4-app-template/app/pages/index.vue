<template>
  <div class="space-y-6">
    <!-- Page Title -->
    <div class="flex items-center justify-between">
      <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">Dashboard</h1>
      <p class="text-sm text-gray-500 dark:text-gray-400">Welcome back, Admin!</p>
    </div>

    <!-- Stats Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <UCard v-for="stat in stats" :key="stat.label" class="p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm text-gray-500 dark:text-gray-400">{{ stat.label }}</p>
            <p class="text-2xl font-bold mt-1">{{ stat.value }}</p>
            <p :class="stat.changeColor" class="text-sm mt-1">
              {{ stat.change > 0 ? '+' : '' }}{{ stat.change }}% from last month
            </p>
          </div>
          <div :class="stat.iconBg" class="w-12 h-12 rounded-lg flex items-center justify-center">
            <UIcon :name="stat.icon" :class="stat.iconColor" class="w-6 h-6" />
          </div>
        </div>
      </UCard>
    </div>

    <!-- Quick Actions & Platform Health -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Quick Actions -->
      <UCard class="lg:col-span-2">
        <template #header>
          <h2 class="text-lg font-semibold">Quick Actions</h2>
        </template>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <UButton
            v-for="action in quickActions"
            :key="action.label"
            :to="action.to"
            :icon="action.icon"
            color="neutral"
            variant="subtle"
            class="flex flex-col items-center justify-center h-24"
          >
            <span class="text-xs mt-2">{{ action.label }}</span>
          </UButton>
        </div>
      </UCard>

      <!-- Platform Health -->
      <UCard>
        <template #header>
          <h2 class="text-lg font-semibold">Platform Health</h2>
        </template>
        <div class="space-y-4">
          <div v-for="metric in platformHealth" :key="metric.label">
            <div class="flex items-center justify-between mb-1">
              <span class="text-sm text-gray-600 dark:text-gray-400">{{ metric.label }}</span>
              <span class="text-sm font-medium">{{ metric.value }}%</span>
            </div>
            <UProgressMeter :value="metric.value" :color="metric.color" size="sm" />
          </div>
        </div>
      </UCard>
    </div>

    <!-- Recent Articles & Recent Activity -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Recent Articles -->
      <UCard>
        <template #header>
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-semibold">Recent Articles</h2>
            <UButton to="/articles" variant="ghost" size="sm">View all</UButton>
          </div>
        </template>
        <UTable :columns="articlesColumns" :rows="recentArticles" class="w-full">
          <template #title="{ row }">
            <div class="flex items-center gap-3">
              <img :src="row.cover" :alt="row.title" class="w-10 h-10 rounded object-cover" />
              <div>
                <p class="font-medium truncate max-w-[200px]">{{ row.title }}</p>
                <p class="text-sm text-gray-500 dark:text-gray-400">{{ row.author }}</p>
              </div>
            </div>
          </template>
          <template #category="{ row }">
            <UBadge :color="getCategoryColor(row.category)" variant="subtle" size="sm">
              {{ row.category }}
            </UBadge>
          </template>
          <template #status="{ row }">
            <UBadge :color="row.status === 'published' ? 'success' : 'warning'" variant="subtle" size="sm">
              {{ row.status }}
            </UBadge>
          </template>
        </UTable>
      </UCard>

      <!-- Recent Activity -->
      <UCard>
        <template #header>
          <h2 class="text-lg font-semibold">Recent Activity</h2>
        </template>
        <div class="space-y-4">
          <div v-for="activity in recentActivity" :key="activity.id" class="flex items-start gap-3">
            <div :class="activity.iconBg" class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0">
              <UIcon :name="activity.icon" :class="activity.iconColor" class="w-4 h-4" />
            </div>
            <div class="flex-1 min-w-0">
              <p class="text-sm">
                <span class="font-medium">{{ activity.user }}</span>
                <span class="text-gray-600 dark:text-gray-400"> {{ activity.action }}</span>
              </p>
              <p class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{{ activity.time }}</p>
            </div>
          </div>
        </div>
      </UCard>
    </div>
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  layout: 'admin'
})

// Stats data
const stats = [
  {
    label: 'Total Users',
    value: '12,846',
    change: 12.5,
    changeColor: 'text-green-600',
    icon: 'i-lucide-users',
    iconColor: 'text-blue-600',
    iconBg: 'bg-blue-100 dark:bg-blue-900'
  },
  {
    label: 'Total Articles',
    value: '1,482',
    change: 8.2,
    changeColor: 'text-green-600',
    icon: 'i-lucide-file-text',
    iconColor: 'text-green-600',
    iconBg: 'bg-green-100 dark:bg-green-900'
  },
  {
    label: 'Premium Members',
    value: '3,254',
    change: 18.3,
    changeColor: 'text-green-600',
    icon: 'i-lucide-crown',
    iconColor: 'text-purple-600',
    iconBg: 'bg-purple-100 dark:bg-purple-900'
  },
  {
    label: 'Revenue',
    value: '$48,296',
    change: -2.4,
    changeColor: 'text-red-600',
    icon: 'i-lucide-dollar-sign',
    iconColor: 'text-green-600',
    iconBg: 'bg-green-100 dark:bg-green-900'
  }
]

// Quick Actions
const quickActions = [
  { label: 'New Article', to: '/articles/create', icon: 'i-lucide-file-plus' },
  { label: 'Add User', to: '/users/create', icon: 'i-lucide-user-plus' },
  { label: 'Create Category', to: '/categories/create', icon: 'i-lucide-folder-plus' },
  { label: 'View Analytics', to: '/analytics', icon: 'i-lucide-bar-chart-2' }
]

// Platform Health
const platformHealth = [
  { label: 'Server Uptime', value: 99.9, color: 'success' },
  { label: 'Response Time', value: 94.2, color: 'primary' },
  { label: 'Error Rate', value: 98.5, color: 'success' }
]

// Articles Table
const articlesColumns = [
  { id: 'title', label: 'Title' },
  { id: 'category', label: 'Category' },
  { id: 'status', label: 'Status' }
]

const recentArticles = [
  {
    id: 1,
    title: 'The Future of AI in Web Development',
    author: 'John Doe',
    category: 'Technology',
    status: 'published',
    cover: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=100&h=100&fit=crop'
  },
  {
    id: 2,
    title: 'Understanding Vue 3 Composition API',
    author: 'Jane Smith',
    category: 'Tutorial',
    status: 'published',
    cover: 'https://images.unsplash.com/photo-1627398242454-45a1465c4b26?w=100&h=100&fit=crop'
  },
  {
    id: 3,
    title: 'Best Practices for Nuxt 4',
    author: 'Mike Johnson',
    category: 'Tutorial',
    status: 'draft',
    cover: 'https://images.unsplash.com/photo-1633356122544-f134324a6cee?w=100&h=100&fit=crop'
  },
  {
    id: 4,
    title: 'Introduction to TypeScript',
    author: 'Sarah Wilson',
    category: 'Beginner',
    status: 'published',
    cover: 'https://images.unsplash.com/photo-1516116216624-53e697fedbea?w=100&h=100&fit=crop'
  }
]

// Recent Activity
const recentActivity = [
  {
    id: 1,
    user: 'John Doe',
    action: 'published a new article',
    time: '5 minutes ago',
    icon: 'i-lucide-file-check',
    iconColor: 'text-green-600',
    iconBg: 'bg-green-100 dark:bg-green-900'
  },
  {
    id: 2,
    user: 'Jane Smith',
    action: 'added a new category',
    time: '1 hour ago',
    icon: 'i-lucide-folder-plus',
    iconColor: 'text-blue-600',
    iconBg: 'bg-blue-100 dark:bg-blue-900'
  },
  {
    id: 3,
    user: 'Mike Johnson',
    action: 'updated user permissions',
    time: '3 hours ago',
    icon: 'i-lucide-user-cog',
    iconColor: 'text-purple-600',
    iconBg: 'bg-purple-100 dark:bg-purple-900'
  },
  {
    id: 4,
    user: 'Sarah Wilson',
    action: 'uploaded 5 new images',
    time: '5 hours ago',
    icon: 'i-lucide-image-plus',
    iconColor: 'text-orange-600',
    iconBg: 'bg-orange-100 dark:bg-orange-900'
  },
  {
    id: 5,
    user: 'System',
    action: 'backup completed successfully',
    time: '1 day ago',
    icon: 'i-lucide-database-backup',
    iconColor: 'text-cyan-600',
    iconBg: 'bg-cyan-100 dark:bg-cyan-900'
  }
]

// Helper functions
const getCategoryColor = (category: string) => {
  const colors: Record<string, any> = {
    Technology: 'blue',
    Tutorial: 'green',
    Beginner: 'cyan',
    Advanced: 'purple'
  }
  return colors[category] || 'neutral'
}
</script>
