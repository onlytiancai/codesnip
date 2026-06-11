<template>
  <NuxtLayout name="default">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Loading State -->
      <div v-if="loading" class="flex justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
      </div>

      <!-- Profile Content -->
      <template v-else-if="profile">
        <!-- Profile Header -->
        <div class="flex flex-col sm:flex-row items-center gap-6 mb-8">
          <UAvatar :src="profile.user.avatar || undefined" :alt="profile.user.name || 'User'" size="3xl" />
          <div class="text-center sm:text-left">
            <h1 class="text-2xl font-bold">{{ profile.user.name || 'User' }}</h1>
            <p class="text-gray-500 dark:text-gray-400">{{ profile.user.email }}</p>
            <div class="flex items-center gap-2 mt-2">
              <UBadge v-if="profile.membership.plan === 'premium'" color="warning" variant="subtle">
                <UIcon name="i-lucide-crown" class="w-3 h-3 mr-1" />
                Premium Member
              </UBadge>
              <UBadge v-else color="neutral" variant="subtle">
                Free Plan
              </UBadge>
              <span class="text-sm text-gray-500">Since {{ formatDate(profile.user.createdAt) }}</span>
            </div>
          </div>
          <div class="sm:ml-auto">
            <UButton to="/user/settings" variant="outline">
              <UIcon name="i-lucide-settings" class="w-4 h-4 mr-2" />
              Edit Profile
            </UButton>
          </div>
        </div>

        <!-- Stats Cards -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          <UCard class="text-center">
            <p class="text-3xl font-bold text-primary">{{ profile.stats.articlesRead }}</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">Articles Read</p>
          </UCard>
          <UCard class="text-center">
            <p class="text-3xl font-bold text-primary">{{ formatReadingTime(profile.stats.totalReadingMinutes) }}</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">Reading Time</p>
          </UCard>
          <UCard class="text-center">
            <p class="text-3xl font-bold text-primary">{{ profile.stats.vocabularyLearned }}</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">Words Learned</p>
          </UCard>
          <UCard class="text-center">
            <p class="text-3xl font-bold text-primary">{{ profile.stats.streak }}</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">Day Streak</p>
          </UCard>
        </div>

        <!-- Membership Card -->
        <UCard v-if="profile.membership.plan === 'premium'" class="mb-8 bg-gradient-to-r from-primary to-primary/80 text-white">
          <div class="flex items-center justify-between">
            <div>
              <h3 class="text-lg font-semibold mb-1">Premium Membership</h3>
              <p class="text-sm opacity-90">Valid until {{ profile.membership.endDate ? formatDate(profile.membership.endDate) : 'N/A' }}</p>
            </div>
            <UButton color="white" variant="soft" to="/membership">
              Manage
            </UButton>
          </div>
        </UCard>
        <UCard v-else class="mb-8 bg-gradient-to-r from-purple-500 to-pink-500 text-white">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-4">
              <UIcon name="i-lucide-crown" class="w-8 h-8" />
              <div>
                <h3 class="font-semibold">Upgrade to Premium</h3>
                <p class="text-sm opacity-90">Unlock unlimited features and advanced tools</p>
              </div>
            </div>
            <UButton color="white" variant="soft" to="/membership">
              Upgrade
            </UButton>
          </div>
        </UCard>

        <!-- Quick Actions -->
        <h2 class="text-lg font-semibold mb-4">Quick Actions</h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
          <NuxtLink to="/user/history">
            <UCard class="hover:border-primary transition cursor-pointer">
              <div class="flex items-center gap-4">
                <div class="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center">
                  <UIcon name="i-lucide-history" class="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h3 class="font-medium">Reading History</h3>
                  <p class="text-sm text-gray-500 dark:text-gray-400">View all articles you've read</p>
                </div>
                <UIcon name="i-lucide-chevron-right" class="w-5 h-5 text-gray-400 ml-auto" />
              </div>
            </UCard>
          </NuxtLink>
          <NuxtLink to="/user/bookmarks">
            <UCard class="hover:border-primary transition cursor-pointer">
              <div class="flex items-center gap-4">
                <div class="w-12 h-12 bg-green-100 dark:bg-green-900 rounded-lg flex items-center justify-center">
                  <UIcon name="i-lucide-bookmark" class="w-6 h-6 text-green-600 dark:text-green-400" />
                </div>
                <div>
                  <h3 class="font-medium">Bookmarks</h3>
                  <p class="text-sm text-gray-500 dark:text-gray-400">Articles you've saved</p>
                </div>
                <UIcon name="i-lucide-chevron-right" class="w-5 h-5 text-gray-400 ml-auto" />
              </div>
            </UCard>
          </NuxtLink>
          <NuxtLink to="/user/vocabulary">
            <UCard class="hover:border-primary transition cursor-pointer">
              <div class="flex items-center gap-4">
                <div class="w-12 h-12 bg-purple-100 dark:bg-purple-900 rounded-lg flex items-center justify-center">
                  <UIcon name="i-lucide-book" class="w-6 h-6 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <h3 class="font-medium">Vocabulary</h3>
                  <p class="text-sm text-gray-500 dark:text-gray-400">Words you're learning</p>
                </div>
                <UIcon name="i-lucide-chevron-right" class="w-5 h-5 text-gray-400 ml-auto" />
              </div>
            </UCard>
          </NuxtLink>
          <NuxtLink to="/user/settings">
            <UCard class="hover:border-primary transition cursor-pointer">
              <div class="flex items-center gap-4">
                <div class="w-12 h-12 bg-orange-100 dark:bg-orange-900 rounded-lg flex items-center justify-center">
                  <UIcon name="i-lucide-settings" class="w-6 h-6 text-orange-600 dark:text-orange-400" />
                </div>
                <div>
                  <h3 class="font-medium">Settings</h3>
                  <p class="text-sm text-gray-500 dark:text-gray-400">Account preferences</p>
                </div>
                <UIcon name="i-lucide-chevron-right" class="w-5 h-5 text-gray-400 ml-auto" />
              </div>
            </UCard>
          </NuxtLink>
        </div>

        <!-- Recent Activity -->
        <h2 class="text-lg font-semibold mb-4">Recent Activity</h2>
        <UCard>
          <div v-if="recentActivity.length > 0" class="space-y-4">
            <div
              v-for="activity in recentActivity"
              :key="activity.id"
              class="flex items-center gap-4 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition"
            >
              <div :class="activity.bgColor" class="w-10 h-10 rounded-lg flex items-center justify-center">
                <UIcon :name="activity.icon" class="w-5 h-5 text-white" />
              </div>
              <div class="flex-1">
                <p class="font-medium">{{ activity.title }}</p>
                <p class="text-sm text-gray-500 dark:text-gray-400">{{ activity.description }}</p>
              </div>
              <span class="text-sm text-gray-400">{{ activity.time }}</span>
            </div>
          </div>
          <div v-else class="text-center py-8">
            <UIcon name="i-lucide-activity" class="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
            <p class="text-gray-500 dark:text-gray-400">No recent activity yet</p>
            <UButton to="/articles" class="mt-4">Start Reading</UButton>
          </div>
        </UCard>
      </template>

      <!-- Not Logged In -->
      <div v-else class="text-center py-12">
        <UIcon name="i-lucide-user" class="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
        <h2 class="text-xl font-semibold mb-2">Please log in</h2>
        <p class="text-gray-500 dark:text-gray-400 mb-4">You need to be logged in to view your profile</p>
        <UButton to="/login">Sign In</UButton>
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  middleware: 'auth'
})

const { profile, loading, fetchProfile } = useUserProfile()
const { history, fetchHistory } = useReadingHistory()

onMounted(async () => {
  await Promise.all([
    fetchProfile(),
    fetchHistory({ limit: 5 })
  ])
})

const recentActivity = computed(() => {
  if (!history.value.length) return []

  return history.value.slice(0, 4).map((item, index) => ({
    id: item.id,
    title: item.progress === 100
      ? `Completed reading "${item.title}"`
      : `Started reading "${item.title}"`,
    description: `${item.readTime} min read • ${item.difficulty}`,
    icon: item.progress === 100 ? 'i-lucide-check-circle' : 'i-lucide-book-open',
    bgColor: item.progress === 100 ? 'bg-green-500' : 'bg-blue-500',
    time: formatRelativeTime(item.lastReadAt)
  }))
})

const formatDate = (date: string | Date) => {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short'
  })
}

const formatReadingTime = (minutes: number) => {
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const mins = minutes % 60
  return `${hours}h ${mins}m`
}

const formatRelativeTime = (date: string | Date) => {
  const now = new Date()
  const then = new Date(date)
  const diffMs = now.getTime() - then.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  return then.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}
</script>