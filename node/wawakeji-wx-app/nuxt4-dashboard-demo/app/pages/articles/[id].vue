<template>
  <NuxtLayout name="default">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Loading State -->
      <div v-if="pending" class="flex justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin" />
      </div>

      <!-- Error State -->
      <div v-else-if="error" class="text-center py-12">
        <UIcon name="i-lucide-file-x" class="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 class="text-lg font-semibold mb-2">Article not found</h3>
        <p class="text-gray-500 dark:text-gray-400 mb-4">{{ error.message }}</p>
        <UButton to="/articles">Back to Articles</UButton>
      </div>

      <template v-else-if="article">
        <!-- Article Header -->
        <div class="mb-8">
          <div class="flex items-center gap-3 mb-4">
            <UBadge color="primary" variant="subtle">{{ article.category?.name }}</UBadge>
            <UBadge :color="difficultyColor(article.difficulty)" variant="subtle">
              {{ capitalize(article.difficulty) }}
            </UBadge>
            <span class="text-sm text-gray-500 dark:text-gray-400">{{ article.readTime }} min read</span>
          </div>
          <h1 class="text-3xl font-bold mb-4">{{ article.title }}</h1>
          <div class="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
            <span>By {{ article.author?.name || 'Unknown' }}</span>
            <span>{{ formatDate(article.createdAt) }}</span>
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
            <span class="text-xs text-gray-500">0:00 / {{ article.readTime }}:00</span>
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
                  <USwitch v-model="showTranslation" />
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm font-medium">Show Phonetics</span>
                  <USwitch v-model="showPhonetics" />
                </div>
                <div class="space-y-2">
                  <span class="text-sm font-medium">Font Size</span>
                  <div class="flex gap-2">
                    <UButton size="xs" variant="outline" @click="updateFontSize(Math.max(12, fontSize - 2))">A-</UButton>
                    <UButton size="xs" variant="outline">{{ fontSize }}</UButton>
                    <UButton size="xs" variant="outline" @click="updateFontSize(Math.min(24, fontSize + 2))">A+</UButton>
                  </div>
                </div>
              </div>
            </template>
          </UPopover>
        </div>

        <!-- Article Content (Sentence by Sentence) -->
        <div class="prose prose-lg dark:prose-invert max-w-none mb-8" :style="{ fontSize: fontSize + 'px' }">
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

        <!-- Tags -->
        <div v-if="article.tags?.length" class="flex flex-wrap gap-2 mb-8">
          <UBadge
            v-for="tag in article.tags"
            :key="tag.id"
            variant="subtle"
          >
            {{ tag.name }}
          </UBadge>
        </div>

        <!-- Actions Bar -->
        <div class="flex items-center justify-between border-t border-gray-200 dark:border-gray-700 pt-6">
          <div class="flex items-center gap-3">
            <UButton
              :icon="isBookmarked ? 'i-lucide-bookmark-check' : 'i-lucide-bookmark'"
              :color="isBookmarked ? 'primary' : 'neutral'"
              variant="soft"
              :loading="bookmarkPending"
              @click="toggleBookmark"
            >
              {{ isBookmarked ? 'Saved' : 'Save' }}
            </UButton>
            <UButton icon="i-lucide-share-2" variant="soft">Share</UButton>
            <UButton icon="i-lucide-book-plus" variant="soft" @click="showVocabModal = true">
              Add Word
            </UButton>
          </div>
          <div class="flex items-center gap-2">
            <UButton icon="i-lucide-arrow-left" variant="ghost" to="/articles">Back</UButton>
          </div>
        </div>
      </template>

      <!-- Add Vocabulary Modal -->
      <UModal v-model:open="showVocabModal">
        <template #content>
          <UCard>
            <template #header>
              <h3 class="text-lg font-semibold">Add to Vocabulary</h3>
            </template>

            <form class="space-y-4" @submit.prevent="handleAddToVocabulary">
              <UFormField label="Word" name="word" required>
                <UInput v-model="selectedWord.word" placeholder="e.g., artificial" />
              </UFormField>
              <UFormField label="Phonetic" name="phonetic">
                <UInput v-model="selectedWord.phonetic" placeholder="e.g., /ˌɑːrtɪˈfɪʃl/" />
              </UFormField>
              <UFormField label="Definition" name="definition" required>
                <UTextarea v-model="selectedWord.definition" placeholder="Word definition..." :rows="2" />
              </UFormField>
            </form>

            <template #footer>
              <div class="flex justify-end gap-2">
                <UButton variant="outline" @click="showVocabModal = false">Cancel</UButton>
                <UButton :loading="addingVocab" @click="handleAddToVocabulary">Add Word</UButton>
              </div>
            </template>
          </UCard>
        </template>
      </UModal>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const route = useRoute()
const { loggedIn } = useUserSession()
const { preferences, fetchPreferences, updatePreferences } = useUserPreferences()
const { addWord } = useVocabulary()
const toast = useToast()

const selectedSentence = ref<string | null>(null)
const showTranslation = ref(true)
const showPhonetics = ref(false)
const isBookmarked = ref(false)
const bookmarkPending = ref(false)
const showVocabModal = ref(false)
const selectedWord = reactive({
  word: '',
  phonetic: '',
  definition: ''
})
const addingVocab = ref(false)

// Font size from preferences
const fontSize = computed(() => preferences.value?.fontSize || 16)

const updateFontSize = async (newSize: number) => {
  if (loggedIn.value) {
    try {
      await updatePreferences({ fontSize: newSize })
    } catch (e) {
      // Ignore errors
    }
  }
}

const slug = computed(() => route.params.id as string)

// Fetch user preferences
onMounted(async () => {
  if (loggedIn.value) {
    try {
      await fetchPreferences()
      // Apply preferences to local state
      showTranslation.value = preferences.value?.showTranslation ?? true
      showPhonetics.value = preferences.value?.showPhonetics ?? false
    } catch (e) {
      // Ignore errors
    }
  }
})

// Fetch article
const { data, pending, error } = await useFetch(`/api/articles/${slug.value}`)
const article = computed(() => data.value)

// Check if article is bookmarked
watchEffect(async () => {
  if (loggedIn.value && article.value) {
    try {
      const bookmarks = await $fetch('/api/user/bookmarks')
      isBookmarked.value = bookmarks.some((b: any) => b.articleId === article.value?.id)
    } catch (e) {
      // Ignore errors
    }
  }
})

// Toggle bookmark
const toggleBookmark = async () => {
  if (!loggedIn.value) {
    toast.add({
      title: 'Please login',
      description: 'You need to login to save articles',
      color: 'warning'
    })
    return
  }

  if (!article.value) return

  bookmarkPending.value = true
  try {
    if (isBookmarked.value) {
      await $fetch(`/api/user/bookmarks/${article.value.id}`, { method: 'DELETE' })
      isBookmarked.value = false
      toast.add({ title: 'Removed from bookmarks', color: 'success' })
    } else {
      await $fetch(`/api/user/bookmarks/${article.value.id}`, { method: 'POST' })
      isBookmarked.value = true
      toast.add({ title: 'Added to bookmarks', color: 'success' })
    }
  } catch (e) {
    toast.add({ title: 'Failed to update bookmark', color: 'error' })
  } finally {
    bookmarkPending.value = false
  }
}

// Track reading progress
watchEffect(() => {
  if (loggedIn.value && article.value) {
    $fetch(`/api/user/history/${article.value.id}`, {
      method: 'POST',
      body: { progress: 0 }
    }).catch(() => {})
  }
})

// Persist showTranslation and showPhonetics when changed
watch([showTranslation, showPhonetics], async ([translation, phonetics]) => {
  if (loggedIn.value && preferences.value) {
    try {
      await updatePreferences({
        showTranslation: translation,
        showPhonetics: phonetics
      })
    } catch (e) {
      // Ignore errors
    }
  }
})

// Add word to vocabulary
const handleAddToVocabulary = async () => {
  if (!selectedWord.word || !selectedWord.definition) {
    toast.add({
      title: 'Missing fields',
      description: 'Word and definition are required',
      color: 'error'
    })
    return
  }

  addingVocab.value = true
  try {
    await addWord({
      word: selectedWord.word,
      phonetic: selectedWord.phonetic || undefined,
      definition: selectedWord.definition,
      articleId: article.value?.id
    })
    toast.add({
      title: 'Word added',
      description: `"${selectedWord.word}" has been added to your vocabulary`,
      color: 'success'
    })
    showVocabModal.value = false
    // Reset form
    selectedWord.word = ''
    selectedWord.phonetic = ''
    selectedWord.definition = ''
  } catch (error: any) {
    toast.add({
      title: 'Failed to add word',
      description: error.data?.message || 'Please try again',
      color: 'error'
    })
  } finally {
    addingVocab.value = false
  }
}

// Open vocabulary modal with selected sentence word
const openVocabModal = (sentence: any) => {
  // For simplicity, use the first word of the sentence
  const words = sentence.en.split(' ')
  if (words.length > 0) {
    selectedWord.word = words[0].replace(/[^\w]/g, '').toLowerCase()
    selectedWord.definition = ''
    selectedWord.phonetic = ''
  }
  showVocabModal.value = true
}

const difficultyColor = (difficulty: string) => {
  switch (difficulty) {
    case 'beginner': return 'success'
    case 'intermediate': return 'warning'
    case 'advanced': return 'error'
    default: return 'neutral'
  }
}

const capitalize = (str: string) => {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : ''
}

const formatDate = (date: string | Date) => {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}

// SEO
useSeoMeta({
  title: () => article.value ? `${article.value.title} - English Reading` : 'Loading...',
  description: () => article.value?.excerpt || ''
})
</script>