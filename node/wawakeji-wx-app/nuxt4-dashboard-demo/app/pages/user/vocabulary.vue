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
            v-model="searchQuery"
            placeholder="Search words..."
            icon="i-lucide-search"
            size="sm"
            class="w-48"
          />
          <UButton variant="outline" size="sm" icon="i-lucide-layers" @click="startFlashcardMode">
            Flashcards
          </UButton>
          <UButton variant="outline" size="sm" icon="i-lucide-plus" @click="showAddModal = true">
            Add Word
          </UButton>
        </div>
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="flex justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
      </div>

      <template v-else>
        <!-- Stats -->
        <div class="grid grid-cols-3 gap-4 mb-8">
          <UCard class="text-center">
            <p class="text-2xl font-bold text-primary">{{ stats?.totalWords || 0 }}</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">Total Words</p>
          </UCard>
          <UCard class="text-center">
            <p class="text-2xl font-bold text-green-500">{{ stats?.mastered || 0 }}</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">Mastered</p>
          </UCard>
          <UCard class="text-center">
            <p class="text-2xl font-bold text-orange-500">{{ stats?.learning || 0 }}</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">Learning</p>
          </UCard>
        </div>

        <!-- Filter -->
        <div class="flex items-center gap-2 mb-6">
          <UButton
            :variant="activeFilter === 'all' ? 'solid' : 'outline'"
            color="primary"
            size="sm"
            @click="setFilter('all')"
          >
            All
          </UButton>
          <UButton
            :variant="activeFilter === 'learning' ? 'solid' : 'outline'"
            size="sm"
            @click="setFilter('learning')"
          >
            Learning
          </UButton>
          <UButton
            :variant="activeFilter === 'mastered' ? 'solid' : 'outline'"
            size="sm"
            @click="setFilter('mastered')"
          >
            Mastered
          </UButton>
          <div class="ml-auto">
            <USelect
              :items="sortOptions"
              :model-value="activeSort"
              size="sm"
              class="w-40"
              @update:model-value="setSort($event as string)"
            />
          </div>
        </div>

        <!-- Word List -->
        <div v-if="filteredVocabulary.length > 0" class="space-y-3">
          <UCard
            v-for="word in filteredVocabulary"
            :key="word.id"
            class="hover:border-primary transition"
          >
            <div class="flex items-start gap-4">
              <div class="flex-1">
                <div class="flex items-center gap-3 mb-1">
                  <h4 class="text-lg font-semibold">{{ word.word }}</h4>
                  <span v-if="word.phonetic" class="text-sm text-gray-500 dark:text-gray-400">
                    {{ word.phonetic }}
                  </span>
                  <UButton icon="i-lucide-volume-2" size="xs" variant="ghost" color="neutral" />
                </div>
                <p class="text-gray-600 dark:text-gray-300 mb-2">{{ word.definition }}</p>
                <p v-if="word.example" class="text-sm text-gray-500 dark:text-gray-400 italic">
                  "{{ word.example }}"
                </p>
                <div class="flex items-center gap-2 mt-3">
                  <UBadge v-if="word.article" color="primary" variant="subtle" size="xs">
                    From: {{ word.article.title }}
                  </UBadge>
                  <span class="text-xs text-gray-400">Added {{ formatRelativeTime(word.createdAt) }}</span>
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
                      :class="word.progress === 100 ? 'text-green-500' : 'text-primary'"
                    />
                  </svg>
                  <span class="absolute inset-0 flex items-center justify-center text-xs font-medium">
                    {{ word.progress }}%
                  </span>
                </div>
                <!-- Action Buttons -->
                <div class="flex items-center gap-1">
                  <UButton
                    v-if="word.progress < 100"
                    icon="i-lucide-check"
                    size="xs"
                    variant="ghost"
                    color="success"
                    title="Mark as Mastered"
                    @click="handleMarkMastered(word.id)"
                  />
                  <UButton
                    v-if="word.progress === 100"
                    icon="i-lucide-refresh-cw"
                    size="xs"
                    variant="ghost"
                    color="warning"
                    title="Mark as Learning"
                    @click="handleMarkLearning(word.id)"
                  />
                  <UButton
                    icon="i-lucide-trash-2"
                    size="xs"
                    variant="ghost"
                    color="error"
                    @click="handleDelete(word.id)"
                  />
                </div>
              </div>
            </div>
          </UCard>
        </div>

        <!-- Empty State -->
        <div v-else class="text-center py-12">
          <UIcon name="i-lucide-book" class="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
          <h3 class="text-lg font-medium mb-2">No vocabulary yet</h3>
          <p class="text-gray-500 dark:text-gray-400 mb-4">
            Start adding words to build your vocabulary
          </p>
          <UButton @click="showAddModal = true">Add Your First Word</UButton>
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

      <!-- Add Word Modal -->
      <UModal v-model:open="showAddModal">
        <template #content>
          <UCard>
            <template #header>
              <h3 class="text-lg font-semibold">Add New Word</h3>
            </template>

            <form class="space-y-4" @submit.prevent="handleAddWord">
              <UFormField label="Word" name="word" required>
                <UInput v-model="newWord.word" placeholder="e.g., artificial" />
              </UFormField>
              <UFormField label="Phonetic" name="phonetic">
                <UInput v-model="newWord.phonetic" placeholder="e.g., /ˌɑːrtɪˈfɪʃl/" />
              </UFormField>
              <UFormField label="Definition" name="definition" required>
                <UTextarea v-model="newWord.definition" placeholder="Word definition..." :rows="2" />
              </UFormField>
              <UFormField label="Example" name="example">
                <UTextarea v-model="newWord.example" placeholder="Example sentence..." :rows="2" />
              </UFormField>
            </form>

            <template #footer>
              <div class="flex justify-end gap-2">
                <UButton variant="outline" @click="showAddModal = false">Cancel</UButton>
                <UButton :loading="adding" @click="handleAddWord">Add Word</UButton>
              </div>
            </template>
          </UCard>
        </template>
      </UModal>

      <!-- Flashcard Modal -->
      <UModal v-model:open="showFlashcard" :ui="{ content: 'max-w-lg' }">
        <template #content>
          <UCard v-if="flashcardWords.length > 0">
            <template #header>
              <div class="flex items-center justify-between">
                <h3 class="text-lg font-semibold">Flashcard Mode</h3>
                <span class="text-sm text-gray-500">{{ currentCardIndex + 1 }} / {{ flashcardWords.length }}</span>
              </div>
            </template>

            <div class="text-center py-8">
              <!-- Front of card -->
              <div v-if="!flashcardFlipped" class="space-y-4">
                <p class="text-3xl font-bold">{{ flashcardWords[currentCardIndex]?.word }}</p>
                <p v-if="flashcardWords[currentCardIndex]?.phonetic" class="text-lg text-gray-500">
                  {{ flashcardWords[currentCardIndex]?.phonetic }}
                </p>
                <UButton variant="outline" @click="flashcardFlipped = true">
                  Show Definition
                </UButton>
              </div>
              <!-- Back of card -->
              <div v-else class="space-y-4">
                <p class="text-2xl font-semibold">{{ flashcardWords[currentCardIndex]?.word }}</p>
                <p v-if="flashcardWords[currentCardIndex]?.phonetic" class="text-gray-500">
                  {{ flashcardWords[currentCardIndex]?.phonetic }}
                </p>
                <p class="text-lg text-gray-700 dark:text-gray-300 border-t border-gray-200 dark:border-gray-700 pt-4">
                  {{ flashcardWords[currentCardIndex]?.definition }}
                </p>
                <p v-if="flashcardWords[currentCardIndex]?.example" class="text-sm text-gray-500 italic">
                  "{{ flashcardWords[currentCardIndex]?.example }}"
                </p>
              </div>
            </div>

            <template #footer>
              <div class="flex justify-between gap-2">
                <UButton
                  variant="outline"
                  :disabled="currentCardIndex === 0"
                  @click="prevCard"
                >
                  Previous
                </UButton>
                <div v-if="flashcardFlipped" class="flex gap-2">
                  <UButton color="error" @click="handleDontKnow">
                    Don't Know
                  </UButton>
                  <UButton color="success" @click="handleKnowIt">
                    Know It
                  </UButton>
                </div>
                <UButton v-else variant="outline" @click="nextCard">
                  Skip
                </UButton>
              </div>
            </template>
          </UCard>
          <UCard v-else>
            <div class="text-center py-8">
              <p class="text-gray-500">No words to review. Add some words first!</p>
            </div>
          </UCard>
        </template>
      </UModal>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  middleware: 'auth'
})

const { vocabulary, stats, pagination, loading, fetchVocabulary, addWord, updateWord, deleteWord } = useVocabulary()
const toast = useToast()

const searchQuery = ref('')
const activeFilter = ref('all')
const activeSort = ref('recent')
const showAddModal = ref(false)
const adding = ref(false)

// Flashcard state
const showFlashcard = ref(false)
const flashcardWords = ref<any[]>([])
const currentCardIndex = ref(0)
const flashcardFlipped = ref(false)

const newWord = reactive({
  word: '',
  phonetic: '',
  definition: '',
  example: ''
})

const sortOptions = [
  { label: 'Recently Added', value: 'recent' },
  { label: 'Alphabetical', value: 'alpha' },
  { label: 'Progress', value: 'progress' }
]

onMounted(() => {
  fetchVocabulary()
})

const filteredVocabulary = computed(() => {
  if (!searchQuery.value) return vocabulary.value

  const query = searchQuery.value.toLowerCase()
  return vocabulary.value.filter(v =>
    v.word.toLowerCase().includes(query) ||
    v.definition.toLowerCase().includes(query)
  )
})

const setFilter = (filter: string) => {
  activeFilter.value = filter
  fetchVocabulary({ filter: filter === 'all' ? undefined : filter, sort: activeSort.value })
}

const setSort = (sort: string) => {
  activeSort.value = sort
  fetchVocabulary({ filter: activeFilter.value === 'all' ? undefined : activeFilter.value, sort })
}

const loadPage = async (page: number) => {
  await fetchVocabulary({
    page,
    filter: activeFilter.value === 'all' ? undefined : activeFilter.value,
    sort: activeSort.value
  })
}

const handleAddWord = async () => {
  if (!newWord.word || !newWord.definition) {
    toast.add({
      title: 'Missing fields',
      description: 'Word and definition are required',
      color: 'error'
    })
    return
  }

  adding.value = true
  try {
    await addWord({
      word: newWord.word,
      phonetic: newWord.phonetic || undefined,
      definition: newWord.definition,
      example: newWord.example || undefined
    })
    toast.add({
      title: 'Word added',
      color: 'success'
    })
    showAddModal.value = false
    // Reset form
    newWord.word = ''
    newWord.phonetic = ''
    newWord.definition = ''
    newWord.example = ''
  } catch (error: any) {
    toast.add({
      title: 'Failed to add word',
      description: error.data?.message || 'Please try again',
      color: 'error'
    })
  } finally {
    adding.value = false
  }
}

const handleDelete = async (id: number) => {
  try {
    await deleteWord(id)
    toast.add({
      title: 'Word removed',
      color: 'success'
    })
  } catch (error) {
    toast.add({
      title: 'Failed to remove word',
      color: 'error'
    })
  }
}

const handleMarkMastered = async (id: number) => {
  try {
    await updateWord(id, { progress: 100 })
    toast.add({
      title: 'Marked as mastered',
      color: 'success'
    })
    // Refresh stats
    fetchVocabulary({ filter: activeFilter.value === 'all' ? undefined : activeFilter.value, sort: activeSort.value })
  } catch (error) {
    toast.add({
      title: 'Failed to update word',
      color: 'error'
    })
  }
}

const handleMarkLearning = async (id: number) => {
  try {
    await updateWord(id, { progress: 50 })
    toast.add({
      title: 'Marked as learning',
      color: 'success'
    })
    // Refresh stats
    fetchVocabulary({ filter: activeFilter.value === 'all' ? undefined : activeFilter.value, sort: activeSort.value })
  } catch (error) {
    toast.add({
      title: 'Failed to update word',
      color: 'error'
    })
  }
}

// Flashcard functions
const startFlashcardMode = () => {
  if (vocabulary.value.length === 0) {
    toast.add({
      title: 'No words to review',
      description: 'Add some words first!',
      color: 'warning'
    })
    return
  }
  // Use words that are not mastered first
  flashcardWords.value = [...vocabulary.value]
    .sort((a, b) => a.progress - b.progress)
    .slice(0, 20) // Limit to 20 cards per session
  currentCardIndex.value = 0
  flashcardFlipped.value = false
  showFlashcard.value = true
}

const nextCard = () => {
  if (currentCardIndex.value < flashcardWords.value.length - 1) {
    currentCardIndex.value++
    flashcardFlipped.value = false
  }
}

const prevCard = () => {
  if (currentCardIndex.value > 0) {
    currentCardIndex.value--
    flashcardFlipped.value = false
  }
}

const handleKnowIt = async () => {
  const word = flashcardWords.value[currentCardIndex.value]
  const newProgress = Math.min(100, word.progress + 20)
  try {
    await updateWord(word.id, { progress: newProgress })
  } catch (e) {
    // Ignore errors
  }
  nextCard()
}

const handleDontKnow = async () => {
  const word = flashcardWords.value[currentCardIndex.value]
  const newProgress = Math.max(0, word.progress - 10)
  try {
    await updateWord(word.id, { progress: newProgress })
  } catch (e) {
    // Ignore errors
  }
  nextCard()
}

const formatRelativeTime = (date: string | Date) => {
  const now = new Date()
  const then = new Date(date)
  const diffMs = now.getTime() - then.getTime()
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffDays < 1) return 'today'
  if (diffDays === 1) return 'yesterday'
  if (diffDays < 7) return `${diffDays} days ago`
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`
  return then.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}
</script>