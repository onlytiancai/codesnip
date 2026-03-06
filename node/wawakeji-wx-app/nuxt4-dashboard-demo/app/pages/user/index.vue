<template>
  <NuxtLayout name="default">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Profile Header -->
      <div class="flex flex-col sm:flex-row items-center gap-6 mb-8">
        <UAvatar src="https://avatars.githubusercontent.com/u/1?v=4" alt="User" size="3xl" />
        <div class="text-center sm:text-left">
          <h1 class="text-2xl font-bold">John Doe</h1>
          <p class="text-gray-500 dark:text-gray-400">john.doe@example.com</p>
          <div class="flex items-center gap-2 mt-2">
            <UBadge color="warning" variant="subtle">
              <UIcon name="i-lucide-crown" class="w-3 h-3 mr-1" />
              Premium Member
            </UBadge>
            <span class="text-sm text-gray-500">Since Jan 2026</span>
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
          <p class="text-3xl font-bold text-primary">156</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Articles Read</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-3xl font-bold text-primary">24h</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Reading Time</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-3xl font-bold text-primary">89</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Words Learned</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-3xl font-bold text-primary">15</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Day Streak</p>
        </UCard>
      </div>

      <!-- Membership Card -->
      <UCard class="mb-8 bg-gradient-to-r from-primary to-primary/80 text-white">
        <div class="flex items-center justify-between">
          <div>
            <h3 class="text-lg font-semibold mb-1">Premium Membership</h3>
            <p class="text-sm opacity-90">Valid until December 31, 2026</p>
          </div>
          <UButton color="white" variant="soft" to="/membership">
            Manage
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
        <div class="space-y-4">
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
      </UCard>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const recentActivity = [
  {
    id: 1,
    title: 'Completed reading "AI in Healthcare"',
    description: '8 minutes • Intermediate',
    icon: 'i-lucide-check-circle',
    bgColor: 'bg-green-500',
    time: '2h ago'
  },
  {
    id: 2,
    title: 'Added 5 new words to vocabulary',
    description: 'artificial, intelligence, diagnosis...',
    icon: 'i-lucide-plus-circle',
    bgColor: 'bg-blue-500',
    time: '3h ago'
  },
  {
    id: 3,
    title: 'Earned "Week Warrior" badge',
    description: 'Read for 7 consecutive days',
    icon: 'i-lucide-award',
    bgColor: 'bg-purple-500',
    time: '1d ago'
  },
  {
    id: 4,
    title: 'Started reading "Climate Change"',
    description: 'Science • Advanced',
    icon: 'i-lucide-book-open',
    bgColor: 'bg-orange-500',
    time: '2d ago'
  }
]
</script>