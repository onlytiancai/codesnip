<template>
  <NuxtLayout name="admin">
    <div class="w-full">
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <div class="flex items-center gap-3">
          <UButton variant="ghost" to="/admin/articles">
            <UIcon name="i-lucide-arrow-left" class="w-4 h-4 mr-2" />
            Back
          </UButton>
          <h2 class="text-2xl font-bold">Edit Article</h2>
        </div>
        <UBadge :color="article?.status === 'published' ? 'success' : 'warning'">
          {{ article?.status }}
        </UBadge>
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="flex items-center justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
      </div>

      <template v-else-if="article">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <!-- Main Editor -->
          <div class="lg:col-span-2 space-y-6">
            <!-- Basic Info -->
            <UCard>
              <template #header>
                <h3 class="font-semibold">Basic Information</h3>
              </template>
              <div class="space-y-4">
                <UFormField label="Title" name="title" required>
                  <UInput v-model="articleForm.title" placeholder="Enter article title" class="w-full" />
                </UFormField>
                <UFormField label="Slug" name="slug" required>
                  <UInput v-model="articleForm.slug" placeholder="article-slug" class="w-full" />
                </UFormField>
                <UFormField label="Excerpt" name="excerpt">
                  <UTextarea v-model="articleForm.excerpt" placeholder="Brief description" :rows="2" class="w-full" />
                </UFormField>
                <UFormField label="Content" name="content" required>
                  <AdminMarkdownEditor
                    v-model="articleForm.content"
                    placeholder="Write or paste your article content..."
                    :rows="15"
                  />
                </UFormField>
              </div>
            </UCard>

            <!-- AI Tools -->
            <UCard>
              <template #header>
                <div class="flex items-center justify-between">
                  <h3 class="font-semibold">AI Processing Tools</h3>
                  <UBadge color="primary" variant="subtle">Batch Processing</UBadge>
                </div>
              </template>
              <div class="space-y-4">
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <UButton
                    variant="outline"
                    block
                    icon="i-lucide-scissors"
                    :disabled="!articleForm.content || processing.split"
                    :loading="processing.split"
                    @click="splitContentAction"
                  >
                    Split Content
                  </UButton>
                  <UButton
                    variant="outline"
                    block
                    icon="i-lucide-languages"
                    :disabled="!hasSplitContent || processing.translate"
                    :loading="processing.translate"
                    @click="translateAll"
                  >
                    Translate All
                  </UButton>
                </div>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <UButton
                    variant="outline"
                    block
                    icon="i-lucide-volume-2"
                    :disabled="!hasSplitContent || processing.tts"
                    :loading="processing.tts"
                    @click="generateTts"
                  >
                    Generate Audio
                  </UButton>
                  <UButton
                    variant="outline"
                    block
                    icon="i-lucide-spell-check"
                    :disabled="!hasSentences || processing.phonetics"
                    :loading="processing.phonetics"
                    @click="generatePhonetics"
                  >
                    Generate Phonetics
                  </UButton>
                </div>
              </div>
            </UCard>

            <!-- Split Content Editor -->
            <UCard v-if="hasSplitContent">
              <template #header>
                <div class="flex items-center justify-between">
                  <h3 class="font-semibold">
                    Split Content
                    <span class="text-sm font-normal text-gray-500 ml-2">
                      ({{ splitContent.paragraphs.length }} paragraphs, {{ splitContent.sentences.length }} sentences)
                    </span>
                  </h3>
                </div>
              </template>

              <!-- Tabs for Paragraphs / Sentences -->
              <UTabs v-model="activeTab" :items="[{ label: 'Paragraphs', value: 0 }, { label: 'Sentences', value: 1 }]" class="mb-4" />

              <!-- Paragraphs View -->
              <div v-if="activeTab === 0" class="space-y-4">
                <div
                  v-for="(paragraph, index) in splitContent.paragraphs"
                  :key="index"
                  class="p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
                >
                  <div class="flex items-center justify-between mb-2">
                    <span class="text-sm font-medium text-gray-500">Paragraph {{ index + 1 }}</span>
                    <div class="flex items-center gap-2">
                      <UBadge v-if="!paragraph.cn" color="warning" variant="subtle">No Translation</UBadge>
                      <UBadge v-if="!paragraph.audio" color="warning" variant="subtle">No Audio</UBadge>
                      <UButton
                        size="xs"
                        variant="ghost"
                        icon="i-lucide-volume-2"
                        :disabled="!paragraph.en"
                        @click="playTts(paragraph.en, paragraph.audio)"
                      />
                    </div>
                  </div>
                  <p class="text-sm mb-2">{{ paragraph.en }}</p>
                  <UInput
                    v-model="paragraph.cn"
                    placeholder="Chinese translation..."
                    class="w-full text-sm"
                  />
                </div>
              </div>

              <!-- Sentences View -->
              <div v-else class="space-y-3 max-h-[600px] overflow-y-auto">
                <div
                  v-for="(sentence, index) in splitContent.sentences"
                  :key="index"
                  class="p-3 border border-gray-200 dark:border-gray-700 rounded-lg"
                >
                  <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center gap-2">
                      <span class="text-xs font-medium text-gray-500">{{ index + 1 }}.</span>
                      <UBadge v-if="sentence.paragraphIndex !== undefined" variant="subtle" size="xs">
                        P{{ sentence.paragraphIndex + 1 }}
                      </UBadge>
                    </div>
                    <div class="flex items-center gap-1">
                      <UBadge v-if="!sentence.cn" color="warning" variant="subtle" size="xs">No CN</UBadge>
                      <UBadge v-if="!sentence.audio" color="warning" variant="subtle" size="xs">No Audio</UBadge>
                      <UBadge v-if="!sentence.phonetics?.length" color="warning" variant="subtle" size="xs">No Phonetics</UBadge>
                      <UButton
                        size="xs"
                        variant="ghost"
                        icon="i-lucide-volume-2"
                        @click="playTts(sentence.en, sentence.audio)"
                      />
                    </div>
                  </div>
                  <p class="text-sm mb-1">{{ sentence.en }}</p>
                  <!-- Phonetics display - vertically aligned with words -->
                  <div v-if="sentence.phonetics?.length" class="flex flex-wrap gap-2 mb-2">
                    <div
                      v-for="(p, pIndex) in sentence.phonetics"
                      :key="pIndex"
                      class="flex flex-col items-center"
                    >
                      <span v-if="p.phonetic" class="text-xs text-gray-400">{{ p.phonetic }}</span>
                      <span v-else class="text-xs text-transparent">.</span>
                      <span class="text-sm">{{ p.text }}</span>
                    </div>
                  </div>
                  <UInput
                    v-model="sentence.cn"
                    placeholder="Chinese translation..."
                    class="w-full text-sm"
                  />
                </div>
              </div>
            </UCard>

            <!-- Legacy Sentences Editor (hidden if splitContent exists) -->
            <UCard v-if="!hasSplitContent">
              <template #header>
                <div class="flex items-center justify-between">
                  <h3 class="font-semibold">Sentences ({{ articleForm.sentences.length }})</h3>
                  <UButton size="sm" variant="outline" icon="i-lucide-plus" @click="addSentence">
                    Add Sentence
                  </UButton>
                </div>
              </template>
              <div class="space-y-4">
                <div
                  v-for="(sentence, index) in articleForm.sentences"
                  :key="index"
                  class="p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
                >
                  <div class="flex items-center justify-between mb-3">
                    <span class="text-sm font-medium text-gray-500">{{ index + 1 }}.</span>
                    <div class="flex items-center gap-1">
                      <UButton size="xs" variant="ghost" icon="i-lucide-volume-2" />
                      <UButton size="xs" variant="ghost" icon="i-lucide-arrow-up" @click="moveSentence(index, -1)" :disabled="index === 0" />
                      <UButton size="xs" variant="ghost" icon="i-lucide-arrow-down" @click="moveSentence(index, 1)" :disabled="index === articleForm.sentences.length - 1" />
                      <UButton size="xs" variant="ghost" color="error" icon="i-lucide-trash-2" @click="removeSentence(index)" />
                    </div>
                  </div>
                  <div class="space-y-2">
                    <div><UInput v-model="sentence.en" placeholder="English sentence" class="w-full" /></div>
                    <div><UInput v-model="sentence.cn" placeholder="Chinese translation" class="w-full" /></div>
                  </div>
                </div>
                <p v-if="articleForm.sentences.length === 0" class="text-center text-gray-500 py-4">
                  No sentences yet. Click "Add Sentence" or use "Split Content" to generate from content.
                </p>
              </div>
            </UCard>
          </div>

          <!-- Sidebar -->
          <div class="space-y-6">
            <!-- Publish Options -->
            <UCard>
              <template #header>
                <h3 class="font-semibold">Publish Options</h3>
              </template>
              <div class="space-y-4">
                <UFormField label="Status" name="status">
                  <USelect
                    v-model="articleForm.status"
                    :items="[
                      { label: 'Draft', value: 'draft' },
                      { label: 'Published', value: 'published' }
                    ]"
                    class="w-full"
                  />
                </UFormField>
                <UFormField label="Last Updated" name="updatedAt">
                  <p class="text-sm text-gray-500">{{ formatDate(article.updatedAt) }}</p>
                </UFormField>
              </div>
            </UCard>

            <!-- Classification -->
            <UCard>
              <template #header>
                <h3 class="font-semibold">Classification</h3>
              </template>
              <div class="space-y-4">
                <UFormField label="Category" name="categoryId">
                  <USelect
                    v-model="articleForm.categoryId"
                    :items="categoryOptions"
                    placeholder="Select category"
                    class="w-full"
                  />
                </UFormField>
                <UFormField label="Difficulty" name="difficulty">
                  <USelect
                    v-model="articleForm.difficulty"
                    :items="[
                      { label: 'Beginner', value: 'beginner' },
                      { label: 'Intermediate', value: 'intermediate' },
                      { label: 'Advanced', value: 'advanced' }
                    ]"
                    class="w-full"
                  />
                </UFormField>
                <UFormField label="Tags" name="tags">
                  <div class="flex flex-wrap gap-2 mb-2">
                    <UBadge v-for="tag in selectedTags" :key="tag.id" color="primary" variant="subtle">
                      {{ tag.name }}
                      <button @click="removeTag(tag.id)" class="ml-1">
                        <UIcon name="i-lucide-x" class="w-3 h-3" />
                      </button>
                    </UBadge>
                  </div>
                  <USelect
                    :items="availableTagOptions"
                    placeholder="Add tag"
                    class="w-full"
                    @update:model-value="addTag"
                  />
                </UFormField>
              </div>
            </UCard>

            <!-- Cover Image -->
            <UCard>
              <template #header>
                <h3 class="font-semibold">Cover Image</h3>
              </template>
              <img
                v-if="articleForm.cover"
                :src="articleForm.cover"
                :alt="articleForm.title"
                class="w-full h-40 object-cover rounded-lg mb-3"
              />
              <!-- Preset images -->
              <div class="grid grid-cols-4 gap-2 mb-3">
                <button
                  v-for="img in presetImages"
                  :key="img"
                  @click="articleForm.cover = img"
                  :class="[
                    'w-full h-16 rounded border-2 overflow-hidden transition',
                    articleForm.cover === img ? 'border-primary' : 'border-transparent hover:border-gray-300'
                  ]"
                >
                  <img :src="img" :alt="img" class="w-full h-full object-cover" />
                </button>
              </div>
              <UFormField name="cover">
                <UInput v-model="articleForm.cover" placeholder="https://example.com/image.jpg" class="w-full" />
              </UFormField>
            </UCard>

            <!-- Stats -->
            <UCard>
              <template #header>
                <h3 class="font-semibold">Article Stats</h3>
              </template>
              <div class="grid grid-cols-2 gap-4 text-center">
                <div>
                  <p class="text-2xl font-bold">{{ article.views }}</p>
                  <p class="text-xs text-gray-500">Views</p>
                </div>
                <div>
                  <p class="text-2xl font-bold">{{ article.bookmarks }}</p>
                  <p class="text-xs text-gray-500">Bookmarks</p>
                </div>
              </div>
            </UCard>

            <!-- Actions -->
            <div class="flex gap-3">
              <UButton variant="outline" class="flex-1" :to="`/articles/${article.slug}`">
                Preview
              </UButton>
              <UButton color="primary" class="flex-1" :loading="saving" @click="saveArticle">
                Save Changes
              </UButton>
            </div>
          </div>
        </div>
      </template>
    </div>

    <!-- Progress Bar Modal -->
    <AdminProgressBar
      :show="progress.show"
      :title="progress.title"
      :current="progress.current"
      :total="progress.total"
      :current-item="progress.currentItem"
      :cancellable="false"
    />

    <!-- Completeness Modal -->
    <AdminCompletenessModal
      v-model:open="completenessModal.open"
      :items="completenessModal.items"
      @save-anyway="handleSaveAnyway"
    />
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false,
  middleware: 'admin'
})

interface SplitContent {
  paragraphs: Array<{
    order: number
    en: string
    cn?: string
    audio?: string
  }>
  sentences: Array<{
    order: number
    paragraphIndex: number
    en: string
    cn?: string
    audio?: string
    phonetics?: Array<{ text: string; word: string; phonetic: string | null }>
  }>
}

const route = useRoute()
const articleId = parseInt(route.params.id as string)
const toast = useToast()

const { article, loading, fetchArticle, updateArticle } = useAdminArticles()
const { categories, fetchCategories } = useAdminCategories()
const { tags, fetchTags } = useAdminTags()

const saving = ref(false)
const selectedTags = ref<{ id: number; name: string }[]>([])
const activeTab = ref(0)

// Preset cover images
const presetImages = [
  '/images/cover-1.jpg',
  '/images/cover-2.jpg',
  '/images/cover-3.jpg',
  '/images/cover-4.jpg'
]

const articleForm = ref({
  title: '',
  slug: '',
  excerpt: '',
  cover: '',
  content: '',
  status: 'draft',
  difficulty: 'beginner',
  categoryId: null as number | null,
  sentences: [] as { order: number; en: string; cn?: string }[]
})

const splitContent = ref<SplitContent>({
  paragraphs: [],
  sentences: []
})

const processing = ref({
  split: false,
  translate: false,
  tts: false,
  phonetics: false
})

const progress = ref({
  show: false,
  title: '',
  current: 0,
  total: 0,
  currentItem: ''
})

const completenessModal = ref({
  open: false,
  items: [] as Array<{ type: 'Paragraph' | 'Sentence'; order: number; missing: string[]; preview?: string }>
})

const hasSplitContent = computed(() => splitContent.value.paragraphs.length > 0 || splitContent.value.sentences.length > 0)
const hasSentences = computed(() => splitContent.value.sentences.length > 0)

const categoryOptions = computed(() =>
  categories.value.map(c => ({ label: c.name, value: c.id }))
)

const availableTagOptions = computed(() =>
  tags.value
    .filter(t => !selectedTags.value.find(st => st.id === t.id))
    .map(t => ({ label: t.name, value: t.id }))
)

const formatDate = (dateStr: string) => {
  return new Date(dateStr).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit'
  })
}

const addTag = (tagId: number) => {
  const tag = tags.value.find(t => t.id === tagId)
  if (tag && !selectedTags.value.find(t => t.id === tagId)) {
    selectedTags.value.push({ id: tag.id, name: tag.name })
  }
}

const removeTag = (tagId: number) => {
  selectedTags.value = selectedTags.value.filter(t => t.id !== tagId)
}

const addSentence = () => {
  articleForm.value.sentences.push({
    order: articleForm.value.sentences.length,
    en: '',
    cn: ''
  })
}

const removeSentence = (index: number) => {
  articleForm.value.sentences.splice(index, 1)
  articleForm.value.sentences.forEach((s, i) => {
    s.order = i
  })
}

const moveSentence = (index: number, direction: number) => {
  const newIndex = index + direction
  if (newIndex < 0 || newIndex >= articleForm.value.sentences.length) return

  const temp = articleForm.value.sentences[index]
  articleForm.value.sentences[index] = articleForm.value.sentences[newIndex]
  articleForm.value.sentences[newIndex] = temp

  articleForm.value.sentences.forEach((s, i) => {
    s.order = i
  })
}

// Play TTS using audio URL or fallback to Web Speech API
const playTts = (text: string, audioUrl?: string) => {
  if (audioUrl) {
    // Use server-generated audio
    const audio = new Audio(audioUrl)
    audio.play().catch(() => {
      // Fallback to speech synthesis if audio fails
      speakWithSynthesis(text)
    })
  } else {
    // Use speech synthesis as fallback
    speakWithSynthesis(text)
  }
}

const speakWithSynthesis = (text: string) => {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = 'en-US'
    utterance.rate = 1.0
    window.speechSynthesis.speak(utterance)
  } else {
    toast.add({ title: 'TTS not supported', color: 'warning' })
  }
}

// Split content into paragraphs and sentences
const splitContentAction = async () => {
  processing.value.split = true
  progress.value = {
    show: true,
    title: 'Splitting Content',
    current: 0,
    total: 1,
    currentItem: 'Processing...'
  }

  try {
    const result = await $fetch(`/api/admin/articles/${articleId}/split`, {
      method: 'POST',
      body: { content: articleForm.value.content }
    })

    splitContent.value = {
      paragraphs: result.paragraphs.map((p: any) => ({ ...p, cn: '', audio: '' })),
      sentences: result.sentences.map((s: any) => ({ ...s, cn: '', audio: '', phonetics: [] }))
    }

    toast.add({ title: 'Content split successfully', color: 'success' })
  } catch (e: any) {
    toast.add({ title: 'Failed to split content', description: e.data?.message, color: 'error' })
  } finally {
    processing.value.split = false
    progress.value.show = false
  }
}

// Translate all items
const translateAll = async () => {
  if (!hasSplitContent.value) return

  processing.value.translate = true
  const total = splitContent.value.paragraphs.length + splitContent.value.sentences.length
  progress.value = {
    show: true,
    title: 'Translating',
    current: 0,
    total,
    currentItem: 'Starting...'
  }

  try {
    // Translate paragraphs first
    if (splitContent.value.paragraphs.length > 0) {
      progress.value.currentItem = 'Translating paragraphs...'
      const paragraphResult = await $fetch(`/api/admin/articles/${articleId}/translate`, {
        method: 'POST',
        body: { paragraphs: splitContent.value.paragraphs }
      })

      paragraphResult.paragraphs.forEach((p: any) => {
        const existing = splitContent.value.paragraphs.find(ep => ep.order === p.order)
        if (existing) existing.cn = p.cn
      })
      progress.value.current = splitContent.value.paragraphs.length
    }

    // Then translate sentences
    if (splitContent.value.sentences.length > 0) {
      progress.value.currentItem = 'Translating sentences...'
      const sentenceResult = await $fetch(`/api/admin/articles/${articleId}/translate`, {
        method: 'POST',
        body: { sentences: splitContent.value.sentences }
      })

      sentenceResult.sentences.forEach((s: any) => {
        const existing = splitContent.value.sentences.find(es => es.order === s.order)
        if (existing) existing.cn = s.cn
      })
      progress.value.current = total
    }

    toast.add({ title: 'Translation completed', color: 'success' })
  } catch (e: any) {
    toast.add({ title: 'Translation failed', description: e.data?.message, color: 'error' })
  } finally {
    processing.value.translate = false
    progress.value.show = false
  }
}

// Generate TTS for all items
const generateTts = async () => {
  if (!hasSplitContent.value) return

  processing.value.tts = true
  const total = splitContent.value.paragraphs.length + splitContent.value.sentences.length
  progress.value = {
    show: true,
    title: 'Generating Audio',
    current: 0,
    total,
    currentItem: 'Starting...'
  }

  try {
    // Generate TTS for paragraphs
    if (splitContent.value.paragraphs.length > 0) {
      progress.value.currentItem = 'Generating audio for paragraphs...'
      const paragraphResult = await $fetch(`/api/admin/articles/${articleId}/tts`, {
        method: 'POST',
        body: { paragraphs: splitContent.value.paragraphs }
      })

      paragraphResult.paragraphs.forEach((p: any) => {
        const existing = splitContent.value.paragraphs.find(ep => ep.order === p.order)
        if (existing) existing.audio = p.audio
      })
      progress.value.current = splitContent.value.paragraphs.length
    }

    // Generate TTS for sentences
    if (splitContent.value.sentences.length > 0) {
      progress.value.currentItem = 'Generating audio for sentences...'
      const sentenceResult = await $fetch(`/api/admin/articles/${articleId}/tts`, {
        method: 'POST',
        body: { sentences: splitContent.value.sentences }
      })

      sentenceResult.sentences.forEach((s: any) => {
        const existing = splitContent.value.sentences.find(es => es.order === s.order)
        if (existing) existing.audio = s.audio
      })
      progress.value.current = total
    }

    toast.add({ title: 'Audio generated (Web Speech API will be used for playback)', color: 'success' })
  } catch (e: any) {
    toast.add({ title: 'TTS generation failed', description: e.data?.message, color: 'error' })
  } finally {
    processing.value.tts = false
    progress.value.show = false
  }
}

// Generate phonetics for sentences
const generatePhonetics = async () => {
  if (!hasSentences.value) return

  processing.value.phonetics = true
  progress.value = {
    show: true,
    title: 'Generating Phonetics',
    current: 0,
    total: splitContent.value.sentences.length,
    currentItem: 'Starting...'
  }

  try {
    const result = await $fetch(`/api/admin/articles/${articleId}/phonetics`, {
      method: 'POST',
      body: { items: splitContent.value.sentences }
    })

    result.items.forEach((item: any) => {
      const existing = splitContent.value.sentences.find(s => s.order === item.order)
      if (existing) {
        existing.phonetics = item.phonetics
      }
      progress.value.current++
      progress.value.currentItem = `Processing sentence ${item.order + 1}`
    })

    toast.add({ title: 'Phonetics generated', color: 'success' })
  } catch (e: any) {
    toast.add({ title: 'Phonetics generation failed', description: e.data?.message, color: 'error' })
  } finally {
    processing.value.phonetics = false
    progress.value.show = false
  }
}

// Check completeness before saving
const checkCompleteness = (): boolean => {
  const incompleteItems: Array<{ type: 'Paragraph' | 'Sentence'; order: number; missing: string[]; preview?: string }> = []

  // Check paragraphs
  splitContent.value.paragraphs.forEach((p, index) => {
    const missing: string[] = []
    if (!p.en) missing.push('English')
    if (!p.cn) missing.push('Translation')
    if (!p.audio) missing.push('Audio')
    if (missing.length > 0) {
      incompleteItems.push({
        type: 'Paragraph',
        order: index,
        missing,
        preview: p.en?.slice(0, 50)
      })
    }
  })

  // Check sentences
  splitContent.value.sentences.forEach((s, index) => {
    const missing: string[] = []
    if (!s.en) missing.push('English')
    if (!s.cn) missing.push('Translation')
    if (!s.audio) missing.push('Audio')
    if (!s.phonetics?.length) missing.push('Phonetics')
    if (missing.length > 0) {
      incompleteItems.push({
        type: 'Sentence',
        order: index,
        missing,
        preview: s.en?.slice(0, 50)
      })
    }
  })

  if (incompleteItems.length > 0) {
    completenessModal.value.items = incompleteItems
    completenessModal.value.open = true
    return false
  }

  return true
}

const pendingSave = ref(false)

const handleSaveAnyway = () => {
  pendingSave.value = true
  saveArticle()
}

const saveArticle = async () => {
  // Check completeness if we have split content
  if (hasSplitContent.value && !pendingSave.value && !checkCompleteness()) {
    return
  }

  saving.value = true
  try {
    const data: any = {
      ...articleForm.value,
      categoryId: articleForm.value.categoryId || null,
      tagIds: selectedTags.value.map(t => t.id),
      sentences: articleForm.value.sentences.map((s, i) => ({
        ...s,
        order: i
      }))
    }

    // Include splitContent if we have it
    if (hasSplitContent.value) {
      data.splitContent = splitContent.value
    }

    await updateArticle(articleId, data)
    toast.add({ title: 'Article saved', color: 'success' })
  } catch (e: any) {
    toast.add({ title: 'Failed to save article', description: e.data?.message, color: 'error' })
  } finally {
    saving.value = false
    pendingSave.value = false
  }
}

// Initialize form with article data
watch(article, (newArticle) => {
  if (newArticle) {
    articleForm.value = {
      title: newArticle.title,
      slug: newArticle.slug,
      excerpt: newArticle.excerpt || '',
      cover: newArticle.cover || '',
      content: newArticle.content || '',
      status: newArticle.status,
      difficulty: newArticle.difficulty,
      categoryId: newArticle.categoryId,
      sentences: newArticle.sentences.map(s => ({
        order: s.order,
        en: s.en,
        cn: s.cn || ''
      }))
    }

    // Load splitContent if exists
    if (newArticle.splitContent) {
      splitContent.value = newArticle.splitContent as SplitContent
    }

    selectedTags.value = newArticle.tags.map(t => ({ id: t.id, name: t.name }))
  }
}, { immediate: true })

onMounted(async () => {
  await Promise.all([
    fetchArticle(articleId),
    fetchCategories(),
    fetchTags()
  ])
})
</script>