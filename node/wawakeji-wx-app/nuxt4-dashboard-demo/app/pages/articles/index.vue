<template>
  <NuxtLayout name="default">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Header -->
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 class="text-2xl font-bold">Articles</h1>
          <p class="text-gray-500 dark:text-gray-400">Browse and discover articles for your level</p>
        </div>
        <UInput
          placeholder="Search articles..."
          icon="i-lucide-search"
          class="w-full sm:w-72"
        />
      </div>

      <!-- Filters -->
      <div class="flex flex-wrap gap-2 mb-6">
        <UButton
          v-for="cat in categoryFilters"
          :key="cat.value"
          :variant="selectedCategory === cat.value ? 'solid' : 'outline'"
          :color="selectedCategory === cat.value ? 'primary' : 'neutral'"
          size="sm"
          @click="selectedCategory = cat.value"
        >
          {{ cat.label }}
        </UButton>
      </div>

      <!-- Difficulty Filter -->
      <div class="flex items-center gap-3 mb-8">
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

      <!-- Articles Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
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
                <UBadge color="primary" variant="subtle" size="xs">{{ article.category }}</UBadge>
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
                <div class="flex items-center gap-1">
                  <UIcon name="i-lucide-bookmark" class="w-3 h-3" />
                  <span>{{ article.bookmarks }}</span>
                </div>
              </div>
            </div>
          </UCard>
        </NuxtLink>
      </div>

      <!-- Pagination -->
      <div class="flex justify-center mt-8">
        <UPagination v-model:page="currentPage" :total="50" :items-per-page="9" />
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const selectedCategory = ref('all')
const selectedDifficulty = ref('all')
const currentPage = ref(1)

const categoryFilters = [
  { label: 'All', value: 'all' },
  { label: 'Technology', value: 'technology' },
  { label: 'Science', value: 'science' },
  { label: 'Business', value: 'business' },
  { label: 'Health', value: 'health' },
  { label: 'Culture', value: 'culture' }
]

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
    category: 'Technology',
    difficulty: 'Intermediate',
    readTime: 8,
    views: '2.3k',
    bookmarks: 156,
    cover: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=400&h=300&fit=crop'
  },
  {
    id: 2,
    title: 'Climate Change: What Scientists Are Saying',
    excerpt: 'Understanding the latest research on global warming and its impacts.',
    category: 'Science',
    difficulty: 'Advanced',
    readTime: 12,
    views: '1.8k',
    bookmarks: 98,
    cover: 'https://images.unsplash.com/photo-1569163139599-0f4517e36f51?w=400&h=300&fit=crop'
  },
  {
    id: 3,
    title: 'Building a Successful Startup: Lessons from Founders',
    excerpt: 'Key insights from entrepreneurs who built billion-dollar companies.',
    category: 'Business',
    difficulty: 'Beginner',
    readTime: 6,
    views: '3.1k',
    bookmarks: 234,
    cover: 'https://images.unsplash.com/photo-1559136555-9303baea8ebd?w=400&h=300&fit=crop'
  },
  {
    id: 4,
    title: 'The Science of Sleep: Why It Matters',
    excerpt: 'Discover how quality sleep affects your health and productivity.',
    category: 'Health',
    difficulty: 'Beginner',
    readTime: 5,
    views: '4.2k',
    bookmarks: 312,
    cover: 'https://images.unsplash.com/photo-1541781774459-bb2af2f05b55?w=400&h=300&fit=crop'
  },
  {
    id: 5,
    title: 'Digital Transformation in Modern Business',
    excerpt: 'How companies are adapting to the digital age with new technologies.',
    category: 'Business',
    difficulty: 'Intermediate',
    readTime: 10,
    views: '1.5k',
    bookmarks: 87,
    cover: 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=400&h=300&fit=crop'
  },
  {
    id: 6,
    title: 'Understanding Quantum Computing',
    excerpt: 'A beginner-friendly introduction to the world of quantum mechanics.',
    category: 'Technology',
    difficulty: 'Advanced',
    readTime: 15,
    views: '892',
    bookmarks: 56,
    cover: 'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=400&h=300&fit=crop'
  },
  {
    id: 7,
    title: 'Traditional Festivals Around the World',
    excerpt: 'Explore unique cultural celebrations from different countries.',
    category: 'Culture',
    difficulty: 'Beginner',
    readTime: 7,
    views: '2.1k',
    bookmarks: 178,
    cover: 'https://images.unsplash.com/photo-1533174072545-7a4b6ad7a6c3?w=400&h=300&fit=crop'
  },
  {
    id: 8,
    title: 'The Psychology of Decision Making',
    excerpt: 'Learn how our brain processes choices and makes decisions.',
    category: 'Science',
    difficulty: 'Intermediate',
    readTime: 9,
    views: '1.9k',
    bookmarks: 145,
    cover: 'https://images.unsplash.com/photo-1559757175-5700dde675bc?w=400&h=300&fit=crop'
  },
  {
    id: 9,
    title: 'Nutrition Myths Debunked',
    excerpt: 'Separating fact from fiction in the world of nutrition science.',
    category: 'Health',
    difficulty: 'Intermediate',
    readTime: 8,
    views: '3.8k',
    bookmarks: 289,
    cover: 'https://images.unsplash.com/photo-1490645935967-10de6ba17061?w=400&h=300&fit=crop'
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