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
            <span v-if="readingTime > 0" class="text-sm text-primary">Reading: {{ readingTime }} min</span>
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

        <!-- Article Content (Sentence by Sentence with Word Hover) -->
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
              <p class="mb-1 leading-relaxed">
                <span
                  v-for="(word, wIndex) in getSentenceWords(sentence.en, pIndex, sIndex)"
                  :key="wIndex"
                  class="word-wrapper inline-block relative"
                  @mouseenter="handleWordHover(word, $event)"
                  @mouseleave="handleWordLeave"
                >
                  <span
                    :class="[
                      'transition-colors duration-150',
                      hoveredWord?.word === word.clean ? 'text-primary bg-primary/10 rounded px-0.5' : ''
                    ]"
                  >
                    <template v-if="showPhonetics && word.phonetic">
                      <ruby class="ruby-text">
                        {{ word.text }}
                        <rt class="text-xs text-gray-500">{{ word.phonetic }}</rt>
                      </ruby>
                    </template>
                    <template v-else>{{ word.text }}</template>
                  </span>
                </span>
              </p>
              <p v-if="showTranslation" class="text-sm text-gray-500 dark:text-gray-400">
                {{ sentence.cn }}
              </p>
            </div>
          </div>
        </div>

        <!-- Word Popup -->
        <Teleport to="body">
          <div
            v-if="showWordPopup && wordPopupData"
            ref="wordPopupRef"
            class="fixed z-50 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 p-4 w-80"
            :style="wordPopupStyle"
            @mouseenter="cancelHidePopup"
            @mouseleave="handleWordLeave"
          >
            <div class="flex items-start justify-between mb-2">
              <div>
                <h4 class="font-semibold text-lg">{{ wordPopupData.word }}</h4>
                <p class="text-sm text-gray-500">{{ wordPopupData.phonetic || '—' }}</p>
              </div>
              <div class="flex items-center gap-1">
                <!-- Audio buttons -->
                <UButton
                  v-if="wordPopupData.audioUs"
                  icon="i-lucide-volume-2"
                  size="xs"
                  color="neutral"
                  variant="ghost"
                  :loading="playingAudioType === 'us'"
                  @click="playAudio('us')"
                >
                  US
                </UButton>
                <UButton
                  v-if="wordPopupData.audioUk"
                  icon="i-lucide-volume-1"
                  size="xs"
                  color="neutral"
                  variant="ghost"
                  :loading="playingAudioType === 'uk'"
                  @click="playAudio('uk')"
                >
                  UK
                </UButton>
                <UButton
                  v-if="loggedIn"
                  icon="i-lucide-plus"
                  size="xs"
                  color="primary"
                  :loading="addingWordToVocab"
                  @click="addWordFromPopup"
                >
                  Add
                </UButton>
              </div>
            </div>

            <!-- Part of speech -->
            <p v-if="wordPopupData.pos" class="text-xs text-primary mb-2">{{ wordPopupData.pos }}</p>

            <!-- English Definition -->
            <div v-if="wordPopupData.definition" class="mb-2">
              <p class="text-xs text-gray-400 mb-0.5">English</p>
              <p class="text-sm text-gray-700 dark:text-gray-300">{{ wordPopupData.definition }}</p>
            </div>

            <!-- Chinese Translation -->
            <div v-if="wordPopupData.translation" class="mb-2">
              <p class="text-xs text-gray-400 mb-0.5">中文</p>
              <p class="text-sm text-gray-700 dark:text-gray-300">{{ wordPopupData.translation }}</p>
            </div>

            <!-- Not found message -->
            <p v-if="!wordPopupData.found && !wordPopupData.definition && !wordPopupData.translation" class="text-sm text-gray-400 italic">
              Word not found in dictionary
            </p>
          </div>
        </Teleport>

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
            <UButton icon="i-lucide-share-2" variant="soft" @click="handleShare">Share</UButton>
          </div>
          <div class="flex items-center gap-2">
            <UButton icon="i-lucide-arrow-left" variant="ghost" to="/articles">Back</UButton>
          </div>
        </div>
      </template>
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

// Word hover state
const hoveredWord = ref<{ word: string; clean: string; text: string } | null>(null)
const showWordPopup = ref(false)
const wordPopupData = ref<{
  word: string
  phonetic: string
  definition: string
  translation: string
  pos?: string
  audioUs?: string
  audioUk?: string
  found?: boolean
} | null>(null)
const wordPopupStyle = ref<Record<string, string>>({})
const hidePopupTimeout = ref<ReturnType<typeof setTimeout> | null>(null)
const addingWordToVocab = ref(false)
const playingAudioType = ref<'us' | 'uk' | null>(null)
const audioRef = ref<HTMLAudioElement | null>(null)

// Reading time tracking
const readingTime = ref(0)
const lastActivityTime = ref(Date.now())
const activityTimeout = ref<ReturnType<typeof setInterval> | null>(null)
const activityListeners = ref<{ name: string; handler: () => void }[]>([])

// Phonetics cache
const phoneticsCache = ref<Map<string, string>>(new Map())

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

// Track user activity
const trackActivity = () => {
  lastActivityTime.value = Date.now()
}

// Start reading time tracking
const startReadingTimeTracking = () => {
  // Add event listeners for user activity
  const events = ['mousemove', 'touchstart', 'scroll', 'keydown'] as const
  events.forEach(eventName => {
    document.addEventListener(eventName, trackActivity)
    activityListeners.value.push({ name: eventName, handler: trackActivity })
  })

  // Update reading time every minute if user was active
  activityTimeout.value = setInterval(() => {
    const now = Date.now()
    const timeSinceLastActivity = now - lastActivityTime.value

    // If user was active in the last minute, increment reading time
    if (timeSinceLastActivity < 60000) {
      readingTime.value++

      // Update progress with reading time
      if (loggedIn.value && article.value) {
        $fetch(`/api/user/history/${article.value.id}`, {
          method: 'POST',
          body: {
            progress: calculateProgress(),
            readingTime: readingTime.value
          }
        }).catch(() => {})
      }
    }
  }, 60000) // Every minute
}

// Calculate reading progress based on scroll position
const calculateProgress = () => {
  const scrollTop = window.scrollY
  const docHeight = document.documentElement.scrollHeight - window.innerHeight
  return Math.min(100, Math.round((scrollTop / docHeight) * 100))
}

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

  // Start reading time tracking
  startReadingTimeTracking()
})

// Cleanup on unmount
onUnmounted(() => {
  if (activityTimeout.value) {
    clearInterval(activityTimeout.value)
  }
  if (hidePopupTimeout.value) {
    clearTimeout(hidePopupTimeout.value)
  }
  // Remove activity listeners
  activityListeners.value.forEach(({ name, handler }) => {
    document.removeEventListener(name, handler)
  })
})

// Fetch article
const { data, pending, error } = await useFetch(`/api/articles/${slug.value}`)
const article = computed(() => data.value)

// Fetch phonetics when article is loaded and showPhonetics is true
watch([() => article.value, showPhonetics], async ([articleData, show]) => {
  if (articleData && show) {
    await fetchPhoneticsForArticle(articleData)
  }
}, { immediate: true })

// Fetch phonetics for all sentences
const fetchPhoneticsForArticle = async (articleData: any) => {
  if (!articleData?.paragraphs) return

  for (const paragraph of articleData.paragraphs) {
    for (const sentence of paragraph.sentences) {
      if (sentence.en) {
        try {
          const result = await $fetch('/api/phonetics', {
            query: { text: sentence.en }
          })
          if (result?.words) {
            for (const w of result.words) {
              phoneticsCache.value.set(w.word.toLowerCase(), w.phonetic)
            }
          }
        } catch (e) {
          // Ignore errors
        }
      }
    }
  }
}

// Get words from a sentence with phonetics
const getSentenceWords = (sentence: string, _pIndex: number, _sIndex: number) => {
  const words: { text: string; clean: string; phonetic?: string }[] = []
  const regex = /(\w+)|([^\w\s]+)/g
  let match

  while ((match = regex.exec(sentence)) !== null) {
    const text = match[0]
    const isWord = !!match[1]
    const clean = text.toLowerCase().replace(/[^\w]/g, '')

    if (isWord) {
      words.push({
        text,
        clean,
        phonetic: phoneticsCache.value.get(clean)
      })
    } else {
      words.push({ text, clean: '' })
    }
  }

  return words
}

// Handle word hover
const handleWordHover = async (word: { text: string; clean: string }, event: MouseEvent) => {
  if (!word.clean) return

  hoveredWord.value = { word: word.clean, clean: word.clean, text: word.text }

  // Cancel any pending hide
  if (hidePopupTimeout.value) {
    clearTimeout(hidePopupTimeout.value)
    hidePopupTimeout.value = null
  }

  // Fetch word data
  try {
    const result = await $fetch('/api/dictionary/lookup', {
      query: { word: word.clean }
    })

    wordPopupData.value = result

    // Position popup
    const target = event.target as HTMLElement
    const rect = target.getBoundingClientRect()

    wordPopupStyle.value = {
      top: `${rect.bottom + 8}px`,
      left: `${Math.min(rect.left, window.innerWidth - 300)}px`
    }

    showWordPopup.value = true
  } catch (e) {
    // Ignore errors
  }
}

// Handle word leave
const handleWordLeave = () => {
  hidePopupTimeout.value = setTimeout(() => {
    showWordPopup.value = false
    hoveredWord.value = null
  }, 200)
}

// Cancel hide popup
const cancelHidePopup = () => {
  if (hidePopupTimeout.value) {
    clearTimeout(hidePopupTimeout.value)
    hidePopupTimeout.value = null
  }
}

// Play audio pronunciation
const playAudio = async (type: 'us' | 'uk') => {
  if (!wordPopupData.value?.audioUs) return

  const url = type === 'us' ? wordPopupData.value.audioUs : wordPopupData.value.audioUk

  // Stop current audio if playing
  if (audioRef.value) {
    audioRef.value.pause()
    audioRef.value = null
  }

  playingAudioType.value = type
  audioRef.value = new Audio(url)

  try {
    await audioRef.value.play()
    audioRef.value.onended = () => {
      playingAudioType.value = null
    }
    audioRef.value.onerror = () => {
      playingAudioType.value = null
    }
  } catch (e) {
    playingAudioType.value = null
  }
}

// Add word from popup
const addWordFromPopup = async () => {
  if (!wordPopupData.value || !loggedIn.value) return

  addingWordToVocab.value = true
  try {
    await addWord({
      word: wordPopupData.value.word,
      phonetic: wordPopupData.value.phonetic,
      definition: wordPopupData.value.translation || wordPopupData.value.definition,
      articleId: article.value?.id
    })
    toast.add({
      title: 'Word added',
      description: `"${wordPopupData.value.word}" has been added to your vocabulary`,
      color: 'success'
    })
  } catch (error: any) {
    toast.add({
      title: 'Failed to add word',
      description: error.data?.message || 'Please try again',
      color: 'error'
    })
  } finally {
    addingWordToVocab.value = false
  }
}

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

// Share article
const handleShare = async () => {
  const url = window.location.href
  const title = article.value?.title || 'Article'

  if (navigator.share) {
    try {
      await navigator.share({
        title,
        url
      })
    } catch (e) {
      // User cancelled or error
    }
  } else {
    // Fallback to clipboard
    try {
      await navigator.clipboard.writeText(url)
      toast.add({
        title: 'Link copied',
        description: 'Article link has been copied to clipboard',
        color: 'success'
      })
    } catch (e) {
      toast.add({
        title: 'Failed to copy',
        description: 'Please copy the URL manually',
        color: 'error'
      })
    }
  }
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

<style scoped>
.word-wrapper {
  margin: 0 1px;
}

.ruby-text rt {
  font-size: 0.6em;
  color: #6b7280;
}
</style>