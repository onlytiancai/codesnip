<template>
  <NuxtLayout name="default">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Category Header -->
      <div class="flex items-center gap-4 mb-8">
        <div :class="category.bgColor" class="w-16 h-16 rounded-xl flex items-center justify-center">
          <UIcon :name="category.icon" class="w-8 h-8 text-white" />
        </div>
        <div>
          <h1 class="text-3xl font-bold">{{ category.name }}</h1>
          <p class="text-gray-500 dark:text-gray-400">{{ category.description }}</p>
        </div>
      </div>

      <!-- Filters -->
      <div class="flex flex-wrap items-center gap-4 mb-8">
        <div class="flex items-center gap-2">
          <span class="text-sm text-gray-500 dark:text-gray-400">Sort by:</span>
          <USelect
            :items="[
              { label: 'Newest', value: 'newest' },
              { label: 'Most Popular', value: 'popular' },
              { label: 'Most Saved', value: 'saved' }
            ]"
            default-value="newest"
            size="sm"
            class="w-36"
          />
        </div>
        <div class="flex items-center gap-2">
          <span class="text-sm text-gray-500 dark:text-gray-400">Difficulty:</span>
          <UBadge
            v-for="diff in difficulties"
            :key="diff.value"
            :color="diff.color"
            :variant="selectedDifficulty === diff.value ? 'solid' : 'subtle'"
            class="cursor-pointer"
            @click="selectedDifficulty = diff.value"
          >
            {{ diff.label }}
          </UBadge>
        </div>
      </div>

      <!-- Articles Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <NuxtLink
          v-for="article in articles"
          :key="article.id"
          :to="`/articles/${article.id}`"
        >
          <UCard class="group h-full hover:border-primary transition cursor-pointer overflow-hidden">
            <img
              :src="article.cover"
              :alt="article.title"
              class="w-full h-48 object-cover group-hover:scale-105 transition duration-300"
            />
            <div class="p-4">
              <div class="flex items-center gap-2 mb-3">
                <UBadge :color="difficultyColor(article.difficulty)" variant="subtle" size="xs">
                  {{ article.difficulty }}
                </UBadge>
              </div>
              <h3 class="font-semibold mb-2 line-clamp-2 group-hover:text-primary transition">
                {{ article.title }}
              </h3>
              <p class="text-sm text-gray-500 dark:text-gray-400 line-clamp-2 mb-4">
                {{ article.excerpt }}
              </p>
              <div class="flex items-center justify-between text-xs text-gray-400">
                <div class="flex items-center gap-1">
                  <UIcon name="i-lucide-clock" class="w-3 h-3" />
                  <span>{{ article.readTime }} min</span>
                </div>
                <div class="flex items-center gap-1">
                  <UIcon name="i-lucide-eye" class="w-3 h-3" />
                  <span>{{ article.views }}</span>
                </div>
              </div>
            </div>
          </UCard>
        </NuxtLink>
      </div>

      <!-- Pagination -->
      <div class="flex justify-center">
        <UPagination v-model:page="currentPage" :total="48" :items-per-page="9" />
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const route = useRoute()
const selectedDifficulty = ref('all')
const currentPage = ref(1)

const category = {
  name: 'Technology',
  slug: 'technology',
  icon: 'i-lucide-cpu',
  description: 'Explore the latest in tech, from AI to software development',
  bgColor: 'bg-blue-500'
}

const difficulties = [
  { label: 'All', value: 'all', color: 'neutral' },
  { label: 'Beginner', value: 'beginner', color: 'success' },
  { label: 'Intermediate', value: 'intermediate', color: 'warning' },
  { label: 'Advanced', value: 'advanced', color: 'error' }
]

const articles = [
  {
    id: 1,
    title: 'The Future of Artificial Intelligence in Healthcare',
    excerpt: 'Explore how AI is revolutionizing medical diagnosis and treatment planning.',
    difficulty: 'Intermediate',
    readTime: 8,
    views: '2.3k',
    cover: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=400&h=300&fit=crop'
  },
  {
    id: 2,
    title: 'Understanding Quantum Computing',
    excerpt: 'A beginner-friendly introduction to the world of quantum mechanics.',
    difficulty: 'Advanced',
    readTime: 15,
    views: '892',
    cover: 'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=400&h=300&fit=crop'
  },
  {
    id: 3,
    title: 'Web Development Trends in 2026',
    excerpt: 'What technologies are shaping the future of web development.',
    difficulty: 'Beginner',
    readTime: 6,
    views: '3.5k',
    cover: 'https://images.unsplash.com/photo-1461749280684-dccba630e2f6?w=400&h=300&fit=crop'
  },
  {
    id: 4,
    title: 'Cybersecurity Best Practices for Everyone',
    excerpt: 'Essential tips to protect yourself online in the digital age.',
    difficulty: 'Beginner',
    readTime: 7,
    views: '4.1k',
    cover: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=400&h=300&fit=crop'
  },
  {
    id: 5,
    title: 'The Rise of Edge Computing',
    excerpt: 'How edge computing is changing data processing and IoT.',
    difficulty: 'Intermediate',
    readTime: 10,
    views: '1.7k',
    cover: 'https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=400&h=300&fit=crop'
  },
  {
    id: 6,
    title: 'Machine Learning vs Deep Learning',
    excerpt: 'Understanding the differences between these AI approaches.',
    difficulty: 'Intermediate',
    readTime: 9,
    views: '2.9k',
    cover: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=400&h=300&fit=crop'
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