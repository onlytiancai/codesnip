<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-950">
    <!-- Navigation -->
    <header class="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex items-center justify-between h-16">
          <div class="flex items-center gap-3">
            <UIcon name="i-lucide-layout-dashboard" class="w-8 h-8 text-primary" />
            <span class="text-xl font-bold">Dashboard</span>
          </div>
          <div class="flex items-center gap-4">
            <UButton icon="i-lucide-bell" color="neutral" variant="ghost" />
            <UAvatar src="https://avatars.githubusercontent.com/u/1?v=4" alt="User" size="sm" />
          </div>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Stats Cards -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <UCard v-for="stat in stats" :key="stat.title">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm text-gray-500 dark:text-gray-400">{{ stat.title }}</p>
              <p class="text-2xl font-bold mt-1">{{ stat.value }}</p>
              <div class="flex items-center gap-1 mt-2">
                <UIcon :name="stat.trend > 0 ? 'i-lucide-trending-up' : 'i-lucide-trending-down'"
                  :class="stat.trend > 0 ? 'text-green-500' : 'text-red-500'" class="w-4 h-4" />
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

      <!-- Tabs Section -->
      <UTabs :items="tabs" class="mb-8">
        <template #overview>
          <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
            <!-- Recent Activity -->
            <UCard class="lg:col-span-2">
              <template #header>
                <h3 class="text-lg font-semibold">Recent Activity</h3>
              </template>
              <div class="space-y-4">
                <div v-for="activity in recentActivity" :key="activity.id"
                  class="flex items-center gap-4 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition">
                  <UAvatar :src="activity.avatar" :alt="activity.user" size="sm" />
                  <div class="flex-1">
                    <p class="font-medium">{{ activity.user }}</p>
                    <p class="text-sm text-gray-500 dark:text-gray-400">{{ activity.action }}</p>
                  </div>
                  <UBadge :color="activity.status === 'completed' ? 'success' : 'warning'" variant="subtle">
                    {{ activity.status }}
                  </UBadge>
                </div>
              </div>
            </UCard>

            <!-- Quick Actions -->
            <UCard>
              <template #header>
                <h3 class="text-lg font-semibold">Quick Actions</h3>
              </template>
              <div class="space-y-3">
                <UButton v-for="action in quickActions" :key="action.label" :icon="action.icon" block
                  variant="soft">
                  {{ action.label }}
                </UButton>
              </div>
            </UCard>
          </div>
        </template>

        <template #analytics>
          <div class="mt-6">
            <UCard>
              <template #header>
                <h3 class="text-lg font-semibold">Analytics Overview</h3>
              </template>
              <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div v-for="metric in analyticsMetrics" :key="metric.label" class="text-center p-4">
                  <UIcon :name="metric.icon" class="w-8 h-8 text-primary mx-auto mb-2" />
                  <p class="text-2xl font-bold">{{ metric.value }}</p>
                  <p class="text-sm text-gray-500 dark:text-gray-400">{{ metric.label }}</p>
                </div>
              </div>
            </UCard>
          </div>
        </template>

        <template #settings>
          <div class="mt-6">
            <UCard>
              <template #header>
                <h3 class="text-lg font-semibold">Settings</h3>
              </template>
              <div class="space-y-4">
                <div class="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                  <div>
                    <p class="font-medium">Dark Mode</p>
                    <p class="text-sm text-gray-500 dark:text-gray-400">Toggle dark mode theme</p>
                  </div>
                  <UButton icon="i-lucide-moon" color="neutral" variant="soft" />
                </div>
                <div class="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                  <div>
                    <p class="font-medium">Notifications</p>
                    <p class="text-sm text-gray-500 dark:text-gray-400">Manage notification preferences</p>
                  </div>
                  <UButton icon="i-lucide-bell" color="neutral" variant="soft" />
                </div>
              </div>
            </UCard>
          </div>
        </template>
      </UTabs>
    </main>
  </div>
</template>

<script setup lang="ts">
const stats = [
  {
    title: 'Total Users',
    value: '12,543',
    trend: 12.5,
    icon: 'i-lucide-users',
    iconBg: 'bg-blue-500'
  },
  {
    title: 'Revenue',
    value: '$45,231',
    trend: 8.2,
    icon: 'i-lucide-dollar-sign',
    iconBg: 'bg-green-500'
  },
  {
    title: 'Orders',
    value: '1,234',
    trend: -3.1,
    icon: 'i-lucide-shopping-cart',
    iconBg: 'bg-purple-500'
  },
  {
    title: 'Growth',
    value: '+24%',
    trend: 4.7,
    icon: 'i-lucide-trending-up',
    iconBg: 'bg-orange-500'
  }
]

const tabs = [
  { label: 'Overview', slot: 'overview' },
  { label: 'Analytics', slot: 'analytics' },
  { label: 'Settings', slot: 'settings' }
]

const recentActivity = [
  {
    id: 1,
    user: 'John Doe',
    action: 'Completed a purchase',
    status: 'completed',
    avatar: 'https://avatars.githubusercontent.com/u/2?v=4'
  },
  {
    id: 2,
    user: 'Jane Smith',
    action: 'Submitted a review',
    status: 'pending',
    avatar: 'https://avatars.githubusercontent.com/u/3?v=4'
  },
  {
    id: 3,
    user: 'Bob Johnson',
    action: 'Updated profile',
    status: 'completed',
    avatar: 'https://avatars.githubusercontent.com/u/4?v=4'
  },
  {
    id: 4,
    user: 'Alice Brown',
    action: 'Created a new order',
    status: 'pending',
    avatar: 'https://avatars.githubusercontent.com/u/5?v=4'
  }
]

const quickActions = [
  { label: 'Add New User', icon: 'i-lucide-user-plus' },
  { label: 'Create Report', icon: 'i-lucide-file-text' },
  { label: 'Send Notification', icon: 'i-lucide-send' }
]

const analyticsMetrics = [
  { label: 'Page Views', value: '54,234', icon: 'i-lucide-eye' },
  { label: 'Sessions', value: '12,345', icon: 'i-lucide-activity' },
  { label: 'Bounce Rate', value: '32%', icon: 'i-lucide-arrow-down-right' }
]
</script>