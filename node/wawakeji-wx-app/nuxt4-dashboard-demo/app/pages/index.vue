<template>
  <NuxtLayout name="default">
    <div>
      <!-- Hero Section -->
      <section class="bg-gradient-to-br from-primary/10 to-primary/5 py-16 sm:py-24">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div class="text-center">
            <h1 class="text-4xl sm:text-5xl font-bold mb-6">
              Master English Through
              <span class="text-primary">Reading</span>
            </h1>
            <p class="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto mb-8">
              Improve your English reading skills with curated articles, interactive vocabulary tools, and personalized progress tracking.
            </p>
            <div class="flex flex-col sm:flex-row gap-4 justify-center">
              <UButton size="lg" to="/articles">
                Start Reading
                <UIcon name="i-lucide-arrow-right" class="ml-2 w-4 h-4" />
              </UButton>
              <UButton size="lg" variant="outline" to="/membership">
                Go Premium
              </UButton>
            </div>
          </div>
        </div>
      </section>

      <!-- Categories Section -->
      <section class="py-16">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 class="text-2xl font-bold mb-8">Browse by Category</h2>
          <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            <NuxtLink
              v-for="category in categories"
              :key="category.slug"
              :to="`/categories/${category.slug}`"
              class="group"
            >
              <UCard class="text-center hover:border-primary transition cursor-pointer">
                <div :class="category.bgColor" class="w-12 h-12 rounded-lg mx-auto mb-3 flex items-center justify-center">
                  <UIcon :name="category.icon" class="w-6 h-6 text-white" />
                </div>
                <h3 class="font-medium group-hover:text-primary transition">{{ category.name }}</h3>
                <p class="text-sm text-gray-500 dark:text-gray-400">{{ category.count }} articles</p>
              </UCard>
            </NuxtLink>
          </div>
        </div>
      </section>

      <!-- Recommended Articles -->
      <section class="py-16 bg-gray-100 dark:bg-gray-900">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div class="flex items-center justify-between mb-8">
            <h2 class="text-2xl font-bold">Recommended for You</h2>
            <NuxtLink to="/articles" class="text-primary hover:underline text-sm font-medium">
              View all
            </NuxtLink>
          </div>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <UCard v-for="article in recommendedArticles" :key="article.id" class="group overflow-hidden">
              <img :src="article.cover" :alt="article.title" class="w-full h-48 object-cover group-hover:scale-105 transition duration-300" />
              <template #footer>
                <div class="p-4">
                  <div class="flex items-center gap-2 mb-2">
                    <UBadge color="primary" variant="subtle" size="xs">{{ article.category }}</UBadge>
                    <UBadge :color="difficultyColor(article.difficulty)" variant="subtle" size="xs">
                      {{ article.difficulty }}
                    </UBadge>
                  </div>
                  <h3 class="font-semibold mb-2 line-clamp-2 group-hover:text-primary transition">
                    {{ article.title }}
                  </h3>
                  <p class="text-sm text-gray-500 dark:text-gray-400 line-clamp-2 mb-3">
                    {{ article.excerpt }}
                  </p>
                  <div class="flex items-center justify-between text-sm text-gray-400">
                    <span>{{ article.readTime }} min read</span>
                    <span>{{ article.views }} views</span>
                  </div>
                </div>
              </template>
            </UCard>
          </div>
        </div>
      </section>

      <!-- Features Section -->
      <section class="py-16">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 class="text-2xl font-bold text-center mb-12">Why Choose Us?</h2>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div class="text-center">
              <div class="w-16 h-16 bg-blue-100 dark:bg-blue-900 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                <UIcon name="i-lucide-headphones" class="w-8 h-8 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 class="text-lg font-semibold mb-2">Audio Narration</h3>
              <p class="text-gray-600 dark:text-gray-400">Listen to native speakers while reading to improve pronunciation.</p>
            </div>
            <div class="text-center">
              <div class="w-16 h-16 bg-green-100 dark:bg-green-900 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                <UIcon name="i-lucide-book-marked" class="w-8 h-8 text-green-600 dark:text-green-400" />
              </div>
              <h3 class="text-lg font-semibold mb-2">Vocabulary Builder</h3>
              <p class="text-gray-600 dark:text-gray-400">Save and review new words with spaced repetition.</p>
            </div>
            <div class="text-center">
              <div class="w-16 h-16 bg-purple-100 dark:bg-purple-900 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                <UIcon name="i-lucide-trending-up" class="w-8 h-8 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 class="text-lg font-semibold mb-2">Progress Tracking</h3>
              <p class="text-gray-600 dark:text-gray-400">Track your reading time, words learned, and streaks.</p>
            </div>
          </div>
        </div>
      </section>

      <!-- CTA Section -->
      <section class="py-16 bg-primary text-white">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 class="text-3xl font-bold mb-4">Ready to Improve Your English?</h2>
          <p class="text-lg opacity-90 mb-8 max-w-2xl mx-auto">
            Join thousands of learners who have improved their reading skills with our platform.
          </p>
          <UButton size="lg" variant="solid" color="white" to="/register">
            Get Started Free
          </UButton>
        </div>
      </section>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const categories = [
  { name: 'Technology', slug: 'technology', icon: 'i-lucide-cpu', count: 42, bgColor: 'bg-blue-500' },
  { name: 'Science', slug: 'science', icon: 'i-lucide-flask-conical', count: 38, bgColor: 'bg-green-500' },
  { name: 'Business', slug: 'business', icon: 'i-lucide-briefcase', count: 56, bgColor: 'bg-purple-500' },
  { name: 'Health', slug: 'health', icon: 'i-lucide-heart-pulse', count: 31, bgColor: 'bg-red-500' },
  { name: 'Culture', slug: 'culture', icon: 'i-lucide-globe', count: 27, bgColor: 'bg-orange-500' },
  { name: 'Travel', slug: 'travel', icon: 'i-lucide-plane', count: 19, bgColor: 'bg-cyan-500' }
]

const recommendedArticles = [
  {
    id: 1,
    title: 'The Future of Artificial Intelligence in Healthcare',
    excerpt: 'Explore how AI is revolutionizing medical diagnosis and treatment planning.',
    category: 'Technology',
    difficulty: 'Intermediate',
    readTime: 8,
    views: '2.3k',
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
    cover: 'https://images.unsplash.com/photo-1559136555-9303baea8ebd?w=400&h=300&fit=crop'
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