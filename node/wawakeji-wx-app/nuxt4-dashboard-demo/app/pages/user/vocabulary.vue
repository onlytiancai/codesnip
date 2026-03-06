<template>
  <NuxtLayout name="default">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Header -->
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 class="text-2xl font-bold">Vocabulary</h1>
          <p class="text-gray-500 dark:text-gray-400">Words you've saved to learn</p>
        </div>
        <div class="flex items-center gap-2">
          <UInput
            placeholder="Search words..."
            icon="i-lucide-search"
            size="sm"
            class="w-48"
          />
          <UButton variant="outline" size="sm" icon="i-lucide-download">
            Export
          </UButton>
        </div>
      </div>

      <!-- Premium Banner -->
      <UCard class="mb-8 bg-gradient-to-r from-green-500 to-teal-500 text-white">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-4">
            <UIcon name="i-lucide-crown" class="w-8 h-8" />
            <div>
              <h3 class="font-semibold">Vocabulary Builder</h3>
              <p class="text-sm opacity-90">Spaced repetition & flashcards with Premium</p>
            </div>
          </div>
          <UButton color="white" variant="soft" to="/membership">
            Upgrade
          </UButton>
        </div>
      </UCard>

      <!-- Stats -->
      <div class="grid grid-cols-3 gap-4 mb-8">
        <UCard class="text-center">
          <p class="text-2xl font-bold text-primary">{{ vocabulary.length }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Total Words</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-green-500">45</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Mastered</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-orange-500">44</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Learning</p>
        </UCard>
      </div>

      <!-- Filter -->
      <div class="flex items-center gap-2 mb-6">
        <UButton variant="solid" color="primary" size="sm">All</UButton>
        <UButton variant="outline" size="sm">Learning</UButton>
        <UButton variant="outline" size="sm">Mastered</UButton>
        <div class="ml-auto">
          <USelect
            :items="[
              { label: 'Recently Added', value: 'recent' },
              { label: 'Alphabetical', value: 'alpha' },
              { label: 'Progress', value: 'progress' }
            ]"
            default-value="recent"
            size="sm"
            class="w-40"
          />
        </div>
      </div>

      <!-- Word List -->
      <div class="space-y-3">
        <UCard
          v-for="word in vocabulary"
          :key="word.id"
          class="hover:border-primary transition"
        >
          <div class="flex items-start gap-4">
            <div class="flex-1">
              <div class="flex items-center gap-3 mb-1">
                <h4 class="text-lg font-semibold">{{ word.word }}</h4>
                <span class="text-sm text-gray-500 dark:text-gray-400">
                  {{ word.phonetic }}
                </span>
                <UButton icon="i-lucide-volume-2" size="xs" variant="ghost" color="neutral" />
              </div>
              <p class="text-gray-600 dark:text-gray-300 mb-2">{{ word.definition }}</p>
              <p class="text-sm text-gray-500 dark:text-gray-400 italic">
                "{{ word.example }}"
              </p>
              <div class="flex items-center gap-2 mt-3">
                <UBadge :color="word.source === 'article' ? 'primary' : 'neutral'" variant="subtle" size="xs">
                  From article
                </UBadge>
                <span class="text-xs text-gray-400">Added {{ word.addedAt }}</span>
              </div>
            </div>
            <div class="flex flex-col items-end gap-2">
              <!-- Progress Ring -->
              <div class="relative w-12 h-12">
                <svg class="w-12 h-12 transform -rotate-90">
                  <circle
                    cx="24"
                    cy="24"
                    r="20"
                    stroke="currentColor"
                    stroke-width="4"
                    fill="none"
                    class="text-gray-200 dark:text-gray-700"
                  />
                  <circle
                    cx="24"
                    cy="24"
                    r="20"
                    stroke="currentColor"
                    stroke-width="4"
                    fill="none"
                    :stroke-dasharray="125.6"
                    :stroke-dashoffset="125.6 - (125.6 * word.progress / 100)"
                    class="text-primary"
                  />
                </svg>
                <span class="absolute inset-0 flex items-center justify-center text-xs font-medium">
                  {{ word.progress }}%
                </span>
              </div>
              <UButton icon="i-lucide-trash-2" size="xs" variant="ghost" color="error" />
            </div>
          </div>
        </UCard>
      </div>

      <!-- Load More -->
      <div class="flex justify-center mt-8">
        <UButton variant="outline">Load More Words</UButton>
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const vocabulary = [
  {
    id: 1,
    word: 'artificial',
    phonetic: '/ˌɑːrtɪˈfɪʃl/',
    definition: 'Made or produced by human beings rather than occurring naturally',
    example: 'Artificial intelligence is transforming many industries.',
    progress: 80,
    source: 'article',
    addedAt: '2 days ago'
  },
  {
    id: 2,
    word: 'diagnosis',
    phonetic: '/ˌdaɪəɡˈnoʊsɪs/',
    definition: 'The identification of the nature of an illness or problem',
    example: 'Early diagnosis is crucial for effective treatment.',
    progress: 60,
    source: 'article',
    addedAt: '3 days ago'
  },
  {
    id: 3,
    word: 'unprecedented',
    phonetic: '/ʌnˈpresɪdentɪd/',
    definition: 'Never done or known before',
    example: 'The pandemic caused unprecedented changes in society.',
    progress: 40,
    source: 'article',
    addedAt: '5 days ago'
  },
  {
    id: 4,
    word: 'revolutionize',
    phonetic: '/ˌrevəˈluːʃənaɪz/',
    definition: 'To change something radically or fundamentally',
    example: 'The internet has revolutionized how we communicate.',
    progress: 100,
    source: 'article',
    addedAt: '1 week ago'
  },
  {
    id: 5,
    word: 'implement',
    phonetic: '/ˈɪmplɪment/',
    definition: 'To put a decision, plan, or agreement into effect',
    example: 'The company plans to implement new policies next month.',
    progress: 50,
    source: 'article',
    addedAt: '1 week ago'
  }
]
</script>