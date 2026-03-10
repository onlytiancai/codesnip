<template>
  <NuxtLayout name="default">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Header -->
      <div class="flex items-center justify-between mb-8">
        <div>
          <h1 class="text-2xl font-bold">Reading History</h1>
          <p class="text-gray-500 dark:text-gray-400">Track your reading progress</p>
        </div>
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="flex justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
      </div>

      <template v-else>
        <!-- Stats Summary -->
        <div class="grid grid-cols-3 gap-4 mb-8">
          <UCard class="text-center">
            <p class="text-2xl font-bold text-primary">{{ stats?.articlesRead || 0 }}</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">Articles Read</p>
          </UCard>
          <UCard class="text-center">
            <p class="text-2xl font-bold text-primary">{{ formatReadingTime(stats?.totalMinutes || 0) }}</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">Total Time</p>
          </UCard>
          <UCard class="text-center">
            <p class="text-2xl font-bold text-primary">{{ stats?.streak || 0 }}</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">Day Streak</p>
          </UCard>
        </div>

        <!-- History List -->
        <div v-if="groupedHistory.length > 0" class="space-y-6">
          <div v-for="group in groupedHistory" :key="group.date">
            <h3 class="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">{{ group.date }}</h3>
            <div class="space-y-3">
              <NuxtLink
                v-for="item in group.items"
                :key="item.id"
                :to="`/articles/${item.slug}`"
              >
                <UCard class="hover:border-primary transition cursor-pointer">
                  <div class="flex items-start gap-4">
                    <img
                      :src="item.cover || '/placeholder.jpg'"
                      :alt="item.title"
                      class="w-20 h-20 object-cover rounded-lg flex-shrink-0 bg-gray-100 dark:bg-gray-800"
                    />
                    <div class="flex-1 min-w-0">
                      <div class="flex items-center gap-2 mb-1">
                        <UBadge v-if="item.category" color="primary" variant="subtle" size="xs">
                          {{ item.category.name }}
                        </UBadge>
                        <UBadge :color="difficultyColor(item.difficulty)" variant="subtle" size="xs">
                          {{ item.difficulty }}
                        </UBadge>
                      </div>
                      <h4 class="font-medium line-clamp-1 mb-1">{{ item.title }}</h4>
                      <div class="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                        <span>{{ item.readTime }} min read</span>
                        <span>{{ item.progress }}% completed</span>
                      </div>
                      <!-- Progress Bar -->
                      <div class="h-1 bg-gray-200 dark:bg-gray-700 rounded-full mt-2 overflow-hidden">
                        <div
                          class="h-full bg-primary rounded-full transition-all"
                          :style="{ width: `${item.progress}%` }"
                        />
                      </div>
                    </div>
                    <div class="flex flex-col items-end gap-2">
                      <span class="text-sm text-gray-400">{{ formatTime(item.lastReadAt) }}</span>
                      <UButton
                        v-if="item.progress < 100"
                        size="xs"
                        color="primary"
                        variant="soft"
                      >
                        Continue
                      </UButton>
                      <UButton
                        v-else
                        size="xs"
                        variant="ghost"
                        icon="i-lucide-check"
                      >
                        Done
                      </UButton>
                    </div>
                  </div>
                </UCard>
              </NuxtLink>
            </div>
          </div>
        </div>

        <!-- Empty State -->
        <div v-else class="text-center py-12">
          <UIcon name="i-lucide-history" class="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
          <h3 class="text-lg font-medium mb-2">No reading history yet</h3>
          <p class="text-gray-500 dark:text-gray-400 mb-4">
            Start reading articles to track your progress
          </p>
          <UButton to="/articles">Browse Articles</UButton>
        </div>

        <!-- Pagination -->
        <div v-if="pagination.totalPages > 1" class="flex justify-center gap-2 mt-8">
          <UButton
            variant="outline"
            size="sm"
            :disabled="pagination.page === 1"
            @click="loadPage(pagination.page - 1)"
          >
            Previous
          </UButton>
          <span class="flex items-center px-4 text-sm text-gray-500">
            Page {{ pagination.page }} of {{ pagination.totalPages }}
          </span>
          <UButton
            variant="outline"
            size="sm"
            :disabled="pagination.page === pagination.totalPages"
            @click="loadPage(pagination.page + 1)"
          >
            Next
          </UButton>
        </div>
      </template>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  middleware: 'auth'
})

const { history, stats, pagination, loading, fetchHistory, fetchStats } = useReadingHistory()

onMounted(async () => {
  await Promise.all([
    fetchHistory(),
    fetchStats()
  ])
})

const groupedHistory = computed(() => {
  const groups: { date: string; items: typeof history.value }[] = []
  const today = new Date()
  today.setHours(0, 0, 0, 0)

  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)

  for (const item of history.value) {
    const itemDate = new Date(item.lastReadAt)
    itemDate.setHours(0, 0, 0, 0)

    let dateLabel: string
    if (itemDate.getTime() === today.getTime()) {
      dateLabel = 'Today'
    } else if (itemDate.getTime() === yesterday.getTime()) {
      dateLabel = 'Yesterday'
    } else {
      dateLabel = itemDate.toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric',
        year: itemDate.getFullYear() !== today.getFullYear() ? 'numeric' : undefined
      })
    }

    let group = groups.find(g => g.date === dateLabel)
    if (!group) {
      group = { date: dateLabel, items: [] }
      groups.push(group)
    }
    group.items.push(item)
  }

  return groups
})

const loadPage = async (page: number) => {
  await fetchHistory({ page })
}

const difficultyColor = (difficulty: string) => {
  switch (difficulty.toLowerCase()) {
    case 'beginner': return 'success'
    case 'intermediate': return 'warning'
    case 'advanced': return 'error'
    default: return 'neutral'
  }
}

const formatReadingTime = (minutes: number) => {
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const mins = minutes % 60
  return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`
}

const formatTime = (date: string | Date) => {
  return new Date(date).toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit'
  })
}
</script>