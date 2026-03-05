<template>
  <div class="min-h-screen bg-gray-50">
    <div v-if="article" class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Article Header -->
      <article class="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <!-- Cover Image -->
        <div v-if="article.coverImage" class="aspect-video bg-gray-100 overflow-hidden">
          <img
            :src="article.coverImage"
            :alt="article.title"
            class="w-full h-full object-cover"
          />
        </div>

        <div class="p-6 md:p-8">
          <!-- Meta Info -->
          <div class="flex items-center flex-wrap gap-3 mb-4">
            <span class="px-3 py-1 bg-blue-100 text-blue-700 text-sm rounded-full">
              {{ getCategoryName(article.category) }}
            </span>
            <span
              class="px-3 py-1 text-sm rounded-full"
              :class="{
                'bg-green-100 text-green-700': article.difficulty === 'beginner',
                'bg-yellow-100 text-yellow-700': article.difficulty === 'intermediate',
                'bg-red-100 text-red-700': article.difficulty === 'advanced',
              }"
            >
              {{ getDifficultyLabel(article.difficulty) }}
            </span>
            <span class="text-sm text-gray-500">
              {{ formatDate(article.publishedAt) }}
            </span>
          </div>

          <!-- Title -->
          <h1 class="text-2xl md:text-3xl font-bold text-gray-900 mb-4">
            {{ article.title }}
          </h1>

          <!-- Summary -->
          <p v-if="article.summary" class="text-gray-600 mb-6">
            {{ article.summary }}
          </p>

          <!-- Actions -->
          <div class="flex items-center gap-4 mb-6">
            <NuxtLink
              to="/articles"
              class="text-gray-600 hover:text-gray-900 text-sm flex items-center gap-1"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
              </svg>
              返回列表
            </NuxtLink>
            <button
              @click="showTranslation = !showTranslation"
              class="text-sm text-blue-600 hover:text-blue-700"
            >
              {{ showTranslation ? '隐藏' : '显示' }}翻译
            </button>
            <button
              @click="showIpa = !showIpa"
              class="text-sm text-blue-600 hover:text-blue-700"
            >
              {{ showIpa ? '隐藏' : '显示' }}音标
            </button>
          </div>

          <!-- Divider -->
          <hr class="border-gray-200 mb-6" />

          <!-- Sentences / Reader -->
          <div class="space-y-4">
            <div
              v-for="sentence in article.sentences"
              :key="sentence.id"
              class="sentence-block p-4 rounded-lg hover:bg-gray-50 transition-colors cursor-pointer"
              @click="playAudio(sentence)"
            >
              <!-- English Sentence with clickable words -->
              <p class="text-lg text-gray-900 mb-2 leading-relaxed">
                <span
                  v-for="(word, index) in sentence.content.split(' ')"
                  :key="index"
                  class="word-segment inline-block px-0.5 py-0.5 rounded hover:bg-yellow-100 cursor-pointer transition-colors"
                  @click.stop="showWordDetail(word)"
                  v-text="word + (index < sentence.content.split(' ').length - 1 ? ' ' : '')"
                ></span>
              </p>

              <!-- IPA -->
              <p v-if="showIpa && sentence.ipa" class="text-sm text-gray-500 mb-1 font-mono">
                /{{ sentence.ipa }}/
              </p>

              <!-- Translation -->
              <p v-if="showTranslation && sentence.translation" class="text-gray-600 text-sm">
                {{ sentence.translation }}
              </p>

              <!-- Audio Indicator -->
              <div v-if="sentence.audioUrl" class="mt-2 flex items-center gap-2 text-xs text-gray-400">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                </svg>
                点击播放
              </div>
            </div>
          </div>
        </div>
      </article>

      <!-- Word Detail Modal -->
      <div
        v-if="selectedWord"
        class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
        @click="selectedWord = null"
      >
        <div
          class="bg-white rounded-xl p-6 max-w-md w-full shadow-xl"
          @click.stop
        >
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-xl font-bold text-gray-900">{{ selectedWord.word }}</h3>
            <button
              @click="selectedWord = null"
              class="text-gray-400 hover:text-gray-600"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div v-if="selectedWord.definition" class="text-gray-600 mb-2">
            {{ selectedWord.definition }}
          </div>
          <div v-if="selectedWord.ipa" class="text-sm text-gray-500 font-mono">
            /{{ selectedWord.ipa }}/
          </div>
          <div class="mt-4 flex gap-2">
            <button class="flex-1 bg-blue-600 hover:bg-blue-700 text-white py-2 rounded-lg text-sm">
              添加到生词本
            </button>
            <button class="flex-1 border border-gray-300 hover:bg-gray-50 text-gray-700 py-2 rounded-lg text-sm">
              发音
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Loading State -->
    <div v-else-if="pending" class="flex items-center justify-center min-h-screen">
      <div class="text-center">
        <div class="inline-block animate-spin rounded-full h-12 w-12 border-4 border-blue-600 border-t-transparent"></div>
        <p class="text-gray-600 mt-4">加载中...</p>
      </div>
    </div>

    <!-- Not Found -->
    <div v-else class="flex items-center justify-center min-h-screen">
      <div class="text-center">
        <div class="text-6xl mb-4">📭</div>
        <h2 class="text-xl font-bold text-gray-900 mb-2">文章不存在</h2>
        <NuxtLink to="/articles" class="text-blue-600 hover:text-blue-700">
          返回文章列表
        </NuxtLink>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ARTICLE_CATEGORIES, formatDate, getDifficultyLabel } from '@wawakeji/shared'

definePageMeta({
  layout: 'default',
})

const route = useRoute()
const slug = route.params.slug as string

// Reader settings
const showTranslation = ref(true)
const showIpa = ref(false)

// Selected word for modal
const selectedWord = ref<{ word: string; definition?: string; ipa?: string } | null>(null)

// Fetch article
const { data: article, pending } = await useFetch(`/api/articles/${slug}`, {
  default: () => null,
})

const getCategoryName = (categoryId: string) => {
  const category = ARTICLE_CATEGORIES.find((c) => c.id === categoryId)
  return category?.name || categoryId
}

// Play audio for a sentence
const playAudio = (sentence: any) => {
  if (!sentence.audioUrl) return

  // TODO: Implement audio playback
  console.log('Playing audio for sentence:', sentence.content)
  // const audio = new Audio(sentence.audioUrl)
  // audio.play()
}

// Show word detail
const showWordDetail = (word: string) => {
  // Clean word (remove punctuation)
  const cleanWord = word.replace(/[.,!?;:"'()]/g, '')
  if (!cleanWord) return

  // TODO: Fetch word definition from API
  selectedWord.value = {
    word: cleanWord,
    definition: '(点击查询单词释义)',
    ipa: '',
  }
}
</script>

<style scoped>
.word-segment:hover {
  background-color: #fef3c7;
}

.sentence-block:hover {
  background-color: #f9fafb;
}
</style>
