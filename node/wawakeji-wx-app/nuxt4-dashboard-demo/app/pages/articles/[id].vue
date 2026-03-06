<template>
  <NuxtLayout name="default">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Article Header -->
      <div class="mb-8">
        <div class="flex items-center gap-3 mb-4">
          <UBadge color="primary" variant="subtle">{{ article.category }}</UBadge>
          <UBadge :color="difficultyColor(article.difficulty)" variant="subtle">
            {{ article.difficulty }}
          </UBadge>
          <span class="text-sm text-gray-500 dark:text-gray-400">{{ article.readTime }} min read</span>
        </div>
        <h1 class="text-3xl font-bold mb-4">{{ article.title }}</h1>
        <div class="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
          <span>By {{ article.author }}</span>
          <span>{{ article.date }}</span>
          <div class="flex items-center gap-1">
            <UIcon name="i-lucide-eye" class="w-4 h-4" />
            <span>{{ article.views }} views</span>
          </div>
        </div>
      </div>

      <!-- Cover Image -->
      <img
        :src="article.cover"
        :alt="article.title"
        class="w-full h-64 sm:h-80 object-cover rounded-xl mb-8"
      />

      <!-- Reading Controls -->
      <div class="flex flex-wrap items-center gap-3 mb-8 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
        <!-- Audio Player -->
        <div class="flex items-center gap-2 flex-1 min-w-0">
          <UButton icon="i-lucide-play" color="primary" size="sm">Play Audio</UButton>
          <div class="flex-1 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div class="h-full w-1/3 bg-primary rounded-full" />
          </div>
          <span class="text-xs text-gray-500">0:00 / 8:30</span>
        </div>

        <!-- Speed Control -->
        <USelect
          :items="[
            { label: '0.5x', value: '0.5' },
            { label: '1.0x', value: '1.0' },
            { label: '1.5x', value: '1.5' },
            { label: '2.0x', value: '2.0' }
          ]"
          default-value="1.0x"
          size="sm"
          class="w-24"
        />

        <!-- Settings -->
        <UPopover>
          <UButton icon="i-lucide-settings" color="neutral" variant="ghost" size="sm" />
          <template #content>
            <div class="p-4 space-y-4 w-64">
              <div class="flex items-center justify-between">
                <span class="text-sm font-medium">Show Translation</span>
                <USwitch default-value />
              </div>
              <div class="flex items-center justify-between">
                <span class="text-sm font-medium">Show Phonetics</span>
                <USwitch />
              </div>
              <div class="space-y-2">
                <span class="text-sm font-medium">Font Size</span>
                <div class="flex gap-2">
                  <UButton size="xs" variant="outline">A-</UButton>
                  <UButton size="xs" variant="outline">A</UButton>
                  <UButton size="xs" variant="outline">A+</UButton>
                </div>
              </div>
            </div>
          </template>
        </UPopover>
      </div>

      <!-- Article Content (Sentence by Sentence) -->
      <div class="prose prose-lg dark:prose-invert max-w-none mb-8">
        <div
          v-for="(paragraph, pIndex) in article.paragraphs"
          :key="pIndex"
          class="mb-6"
        >
          <div
            v-for="(sentence, sIndex) in paragraph.sentences"
            :key="sIndex"
            :class="[
              'cursor-pointer transition rounded px-1 -mx-1',
              selectedSentence === `${pIndex}-${sIndex}`
                ? 'bg-primary/10 ring-2 ring-primary'
                : 'hover:bg-gray-100 dark:hover:bg-gray-800'
            ]"
            @click="selectedSentence = `${pIndex}-${sIndex}`"
          >
            <p class="mb-1">{{ sentence.en }}</p>
            <p v-if="showTranslation" class="text-sm text-gray-500 dark:text-gray-400">
              {{ sentence.cn }}
            </p>
          </div>
        </div>
      </div>

      <!-- Word Tooltip (shown when word is clicked) -->
      <div v-if="selectedWord" class="fixed bottom-4 left-1/2 -translate-x-1/2 z-50">
        <UCard class="w-80">
          <div class="flex items-start justify-between mb-2">
            <div>
              <span class="font-bold text-lg">{{ selectedWord.word }}</span>
              <span class="text-sm text-gray-500 dark:text-gray-400 ml-2">
                {{ selectedWord.phonetic }}
              </span>
            </div>
            <UButton icon="i-lucide-plus" size="xs" color="primary" variant="ghost">
              Add to Vocabulary
            </UButton>
          </div>
          <p class="text-sm text-gray-600 dark:text-gray-300">{{ selectedWord.definition }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400 mt-2">{{ selectedWord.example }}</p>
          <div class="flex gap-2 mt-3">
            <UButton icon="i-lucide-volume-2" size="xs" variant="ghost" />
            <UButton icon="i-lucide-bookmark" size="xs" variant="ghost" />
          </div>
        </UCard>
      </div>

      <!-- Progress Bar -->
      <div class="fixed bottom-0 left-0 right-0 h-1 bg-gray-200 dark:bg-gray-800">
        <div class="h-full w-1/3 bg-primary transition-all duration-300" />
      </div>

      <!-- Actions Bar -->
      <div class="flex items-center justify-between border-t border-gray-200 dark:border-gray-700 pt-6">
        <div class="flex items-center gap-3">
          <UButton icon="i-lucide-bookmark" variant="soft">Save</UButton>
          <UButton icon="i-lucide-share-2" variant="soft">Share</UButton>
        </div>
        <div class="flex items-center gap-2">
          <UButton icon="i-lucide-arrow-left" variant="ghost">Previous</UButton>
          <UButton icon="i-lucide-arrow-right" color="primary">
            Next Article
          </UButton>
        </div>
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const route = useRoute()
const selectedSentence = ref<string | null>(null)
const selectedWord = ref<any>(null)
const showTranslation = ref(true)

const article = {
  id: route.params.id,
  title: 'The Future of Artificial Intelligence in Healthcare',
  category: 'Technology',
  difficulty: 'Intermediate',
  readTime: 8,
  author: 'Sarah Johnson',
  date: 'March 5, 2026',
  views: '2.3k',
  cover: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=800&h=400&fit=crop',
  paragraphs: [
    {
      sentences: [
        {
          en: 'Artificial intelligence is transforming the healthcare industry in unprecedented ways.',
          cn: '人工智能正在以前所未有的方式改变医疗行业。'
        },
        {
          en: 'From diagnostic tools to personalized treatment plans, AI is revolutionizing patient care.',
          cn: '从诊断工具到个性化治疗方案，人工智能正在彻底改变患者护理。'
        }
      ]
    },
    {
      sentences: [
        {
          en: 'One of the most promising applications is in medical imaging analysis.',
          cn: '最有前途的应用之一是医学影像分析。'
        },
        {
          en: 'Machine learning algorithms can now detect certain cancers with higher accuracy than human radiologists.',
          cn: '机器学习算法现在可以比人类放射科医生更准确地检测某些癌症。'
        }
      ]
    },
    {
      sentences: [
        {
          en: 'However, challenges remain in implementing these technologies across healthcare systems.',
          cn: '然而，在医疗系统中实施这些技术仍然存在挑战。'
        },
        {
          en: 'Privacy concerns, regulatory hurdles, and the need for large datasets continue to pose obstacles.',
          cn: '隐私问题、监管障碍以及对大型数据集的需求仍然是障碍。'
        }
      ]
    }
  ]
}

const difficultyColor = (difficulty: string) => {
  switch (difficulty) {
    case 'Beginner': return 'success'
    case 'Intermediate': return 'warning'
    case 'Advanced': return 'error'
    default: return 'neutral'
  }
}
</script>