<template>
  <NuxtLayout name="default">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Header -->
      <div class="flex items-center justify-between mb-8">
        <div>
          <h1 class="text-2xl font-bold">Reading History</h1>
          <p class="text-gray-500 dark:text-gray-400">Track your reading progress</p>
        </div>
        <USelect
          :items="[
            { label: 'All Time', value: 'all' },
            { label: 'This Week', value: 'week' },
            { label: 'This Month', value: 'month' }
          ]"
          default-value="all"
          size="sm"
          class="w-36"
        />
      </div>

      <!-- Stats Summary -->
      <div class="grid grid-cols-3 gap-4 mb-8">
        <UCard class="text-center">
          <p class="text-2xl font-bold text-primary">156</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Articles Read</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-primary">24h 35m</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Total Time</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-primary">15</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Day Streak</p>
        </UCard>
      </div>

      <!-- History List -->
      <div class="space-y-4">
        <div v-for="group in historyGroups" :key="group.date">
          <h3 class="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">{{ group.date }}</h3>
          <div class="space-y-3">
            <NuxtLink
              v-for="item in group.items"
              :key="item.id"
              :to="`/articles/${item.id}`"
            >
              <UCard class="hover:border-primary transition cursor-pointer">
                <div class="flex items-start gap-4">
                  <img
                    :src="item.cover"
                    :alt="item.title"
                    class="w-20 h-20 object-cover rounded-lg flex-shrink-0"
                  />
                  <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 mb-1">
                      <UBadge color="primary" variant="subtle" size="xs">{{ item.category }}</UBadge>
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
                    <span class="text-sm text-gray-400">{{ item.time }}</span>
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

      <!-- Load More -->
      <div class="flex justify-center mt-8">
        <UButton variant="outline">Load More</UButton>
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const historyGroups = [
  {
    date: 'Today',
    items: [
      {
        id: 1,
        title: 'The Future of Artificial Intelligence in Healthcare',
        category: 'Technology',
        difficulty: 'Intermediate',
        readTime: 8,
        progress: 100,
        time: '2:30 PM',
        cover: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=100&h=100&fit=crop'
      },
      {
        id: 2,
        title: 'Climate Change: What Scientists Are Saying',
        category: 'Science',
        difficulty: 'Advanced',
        readTime: 12,
        progress: 45,
        time: '11:15 AM',
        cover: 'https://images.unsplash.com/photo-1569163139599-0f4517e36f51?w=100&h=100&fit=crop'
      }
    ]
  },
  {
    date: 'Yesterday',
    items: [
      {
        id: 3,
        title: 'Building a Successful Startup: Lessons from Founders',
        category: 'Business',
        difficulty: 'Beginner',
        readTime: 6,
        progress: 100,
        time: '8:45 PM',
        cover: 'https://images.unsplash.com/photo-1559136555-9303baea8ebd?w=100&h=100&fit=crop'
      },
      {
        id: 4,
        title: 'The Science of Sleep: Why It Matters',
        category: 'Health',
        difficulty: 'Beginner',
        readTime: 5,
        progress: 100,
        time: '3:20 PM',
        cover: 'https://images.unsplash.com/photo-1541781774459-bb2af2f05b55?w=100&h=100&fit=crop'
      }
    ]
  },
  {
    date: 'March 4, 2026',
    items: [
      {
        id: 5,
        title: 'Understanding Quantum Computing',
        category: 'Technology',
        difficulty: 'Advanced',
        readTime: 15,
        progress: 30,
        time: '10:00 AM',
        cover: 'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=100&h=100&fit=crop'
      }
    ]
  }
]

const difficultyColor = (difficulty: string) => {
  switch (difficulty) {
    case 'Beginner': return 'success'
    case 'Intermediate': return 'warning'
    case 'Advanced': return 'error'
    default: return 'neutral'
  }
}
</script>