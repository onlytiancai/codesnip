<template>
  <div>
    <!-- Hero Section -->
    <section class="py-20 bg-gradient-to-b from-blue-50 to-white">
      <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h1 class="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
          提升技术英语阅读能力
          <span class="block text-blue-600 mt-2">从程序员英语开始</span>
        </h1>
        <p class="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
          专为程序员设计的英语阅读学习应用，通过阅读技术文章、逐句跟读练习，
          沉浸式提升英语阅读和口语能力。
        </p>
        <div class="flex justify-center space-x-4">
          <NuxtLink
            to="/articles"
            class="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg text-lg font-medium transition-colors"
          >
            开始阅读
          </NuxtLink>
          <NuxtLink
            to="/about"
            class="bg-white hover:bg-gray-50 text-gray-700 px-8 py-3 rounded-lg text-lg font-medium border border-gray-300 transition-colors"
          >
            了解更多
          </NuxtLink>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="py-16 bg-white">
      <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 class="text-3xl font-bold text-center text-gray-900 mb-12">
          核心功能
        </h2>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          <div class="text-center p-6">
            <div class="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <svg class="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            </div>
            <h3 class="text-lg font-semibold text-gray-900 mb-2">沉浸式阅读</h3>
            <p class="text-gray-600 text-sm">精选技术文章，营造纯英文阅读环境，循序渐进提升阅读能力</p>
          </div>

          <div class="text-center p-6">
            <div class="w-16 h-16 bg-green-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <svg class="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </div>
            <h3 class="text-lg font-semibold text-gray-900 mb-2">逐句跟读</h3>
            <p class="text-gray-600 text-sm">AI 朗读示范，录音对比练习，实时发音反馈</p>
          </div>

          <div class="text-center p-6">
            <div class="w-16 h-16 bg-purple-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <svg class="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <h3 class="text-lg font-semibold text-gray-900 mb-2">单词即点即查</h3>
            <p class="text-gray-600 text-sm">点击单词即可查看释义和音标，无需切换词典</p>
          </div>

          <div class="text-center p-6">
            <div class="w-16 h-16 bg-orange-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <svg class="w-8 h-8 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
              </svg>
            </div>
            <h3 class="text-lg font-semibold text-gray-900 mb-2">文章分享</h3>
            <p class="text-gray-600 text-sm">生成分享卡片，与好友一起学习进步</p>
          </div>
        </div>
      </div>
    </section>

    <!-- Categories Section -->
    <section class="py-16 bg-gray-50">
      <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 class="text-3xl font-bold text-center text-gray-900 mb-4">
          文章分类
        </h2>
        <p class="text-gray-600 text-center mb-12">
          涵盖前端、后端、移动开发等多个技术领域
        </p>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <NuxtLink
            v-for="category in categories"
            :key="category.id"
            :to="`/articles?category=${category.id}`"
            class="bg-white hover:bg-gray-50 border border-gray-200 rounded-xl p-6 text-center transition-all hover:shadow-md"
          >
            <div class="text-2xl mb-2">{{ category.emoji }}</div>
            <div class="font-medium text-gray-900">{{ category.name }}</div>
            <div class="text-sm text-gray-500 mt-1">{{ category.count }} 篇文章</div>
          </NuxtLink>
        </div>
      </div>
    </section>

    <!-- Latest Articles Section -->
    <section class="py-16 bg-white">
      <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center mb-8">
          <h2 class="text-3xl font-bold text-gray-900">
            最新文章
          </h2>
          <NuxtLink to="/articles" class="text-blue-600 hover:text-blue-700 font-medium">
            查看全部 →
          </NuxtLink>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <NuxtLink
            v-for="article in latestArticles"
            :key="article.id"
            :to="`/articles/${article.slug}`"
            class="bg-white border border-gray-200 rounded-xl overflow-hidden hover:shadow-lg transition-all group"
          >
            <div v-if="article.coverImage" class="aspect-video bg-gray-100 overflow-hidden">
              <img
                :src="article.coverImage"
                :alt="article.title"
                class="w-full h-full object-cover group-hover:scale-105 transition-transform"
              />
            </div>
            <div v-else class="aspect-video bg-gradient-to-br from-blue-100 to-purple-100 flex items-center justify-center">
              <span class="text-4xl">📝</span>
            </div>
            <div class="p-4">
              <div class="flex items-center space-x-2 mb-2">
                <span class="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full">
                  {{ getCategoryName(article.category) }}
                </span>
                <span class="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                  {{ getDifficultyLabel(article.difficulty) }}
                </span>
              </div>
              <h3 class="font-semibold text-gray-900 mb-2 line-clamp-2 group-hover:text-blue-600 transition-colors">
                {{ article.title }}
              </h3>
              <p class="text-gray-600 text-sm line-clamp-2 mb-3">
                {{ article.summary }}
              </p>
              <div class="text-xs text-gray-500">
                {{ formatDate(article.publishedAt) }}
              </div>
            </div>
          </NuxtLink>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="py-16 bg-gradient-to-r from-blue-600 to-blue-700">
      <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h2 class="text-3xl font-bold text-white mb-4">
          开始你的英语学习之旅
        </h2>
        <p class="text-blue-100 mb-8 text-lg">
          注册账户，保存学习进度，追踪学习成果
        </p>
        <NuxtLink
          to="/register"
          class="bg-white hover:bg-gray-50 text-blue-600 px-8 py-3 rounded-lg text-lg font-medium transition-colors inline-block"
        >
          免费注册
        </NuxtLink>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ARTICLE_CATEGORIES, formatDate, getDifficultyLabel } from '@wawakeji/shared'

definePageMeta({
  layout: 'default',
})

const { data: categories } = await useFetch('/api/categories', {
  default: () => [],
  transform: (data) => {
    const categoryMap: Record<string, string> = {
      frontend: '🎨',
      backend: '⚙️',
      mobile: '📱',
      devops: '🚀',
      ai: '🤖',
      database: '🗄️',
      tools: '🛠️',
      career: '💼',
    }
    return data.map((cat: any) => ({
      ...cat,
      emoji: categoryMap[cat.id] || '📄',
    }))
  },
})

const { data: latestArticles } = await useFetch('/api/articles', {
  query: { pageSize: 6 },
  default: () => ({ items: [] }),
  transform: (data) => data.items,
})

const getCategoryName = (categoryId: string) => {
  const category = ARTICLE_CATEGORIES.find((c) => c.id === categoryId)
  return category?.name || categoryId
}
</script>
