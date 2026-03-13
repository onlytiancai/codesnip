<template>
  <NuxtLayout name="admin">
    <div class="w-full">
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-2xl font-bold">Create Article</h2>
        <UButton variant="ghost" to="/admin/articles">
          <UIcon name="i-lucide-arrow-left" class="w-4 h-4 mr-2" />
          Back to List
        </UButton>
      </div>

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
                <UInput v-model="articleForm.title" placeholder="Enter article title" class="w-full" @input="generateSlug" />
              </UFormField>
              <UFormField label="Slug" name="slug" required>
                <UInput v-model="articleForm.slug" placeholder="article-slug" class="w-full" />
              </UFormField>
              <UFormField label="Excerpt" name="excerpt">
                <UTextarea v-model="articleForm.excerpt" placeholder="Brief description of the article" :rows="2" class="w-full" />
              </UFormField>
              <UFormField label="Content" name="content" required>
                <UTextarea
                  v-model="articleForm.content"
                  placeholder="Write or paste your article content here..."
                  :rows="15"
                  class="w-full"
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
            <UTabs v-model="activeTab" :items="[{ label: 'Paragraphs' }, { label: 'Sentences' }]" class="mb-4" />

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
                      @click="playTts(paragraph.en)"
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
                      @click="playTts(sentence.en)"
                    />
                  </div>
                </div>
                <p class="text-sm mb-1">{{ sentence.en }}</p>
                <!-- Phonetics display -->
                <div v-if="sentence.phonetics?.length" class="flex flex-wrap gap-1 mb-2">
                  <span
                    v-for="(p, pIndex) in sentence.phonetics"
                    :key="pIndex"
                    class="text-xs text-gray-500"
                  >
                    {{ p.word }} {{ p.phonetic }}
                  </span>
                </div>
                <UInput
                  v-model="sentence.cn"
                  placeholder="Chinese translation..."
                  class="w-full text-sm"
                />
              </div>
            </div>
          </UCard>

          <!-- SEO Settings -->
          <UCard>
            <template #header>
              <h3 class="font-semibold">SEO Settings</h3>
            </template>
            <div class="space-y-4">
              <UFormField label="Meta Title" name="metaTitle">
                <UInput v-model="articleForm.metaTitle" placeholder="SEO title (optional)" class="w-full" />
              </UFormField>
              <UFormField label="Meta Description" name="metaDesc">
                <UTextarea v-model="articleForm.metaDesc" placeholder="SEO description (optional)" :rows="2" class="w-full" />
              </UFormField>
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
              <UFormField label="Publish Date" name="publishAt">
                <UInput v-model="articleForm.publishAt" type="datetime-local" class="w-full" />
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
                  placeholder="Select difficulty"
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
            <UFormField name="cover">
              <UInput v-model="articleForm.cover" placeholder="https://example.com/image.jpg" class="w-full" />
            </UFormField>
          </UCard>

          <!-- Actions -->
          <div class="flex gap-3">
            <UButton variant="outline" class="flex-1" :loading="saving" @click="saveDraft">
              Save Draft
            </UButton>
            <UButton color="primary" class="flex-1" :loading="saving" @click="publish">
              {{ articleForm.status === 'published' ? 'Publish' : 'Save' }}
            </UButton>
          </div>
        </div>
      </div>
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
    phonetics?: Array<{ word: string; phonetic: string }>
  }>
}

const { createArticle, loading } = useAdminArticles()
const { categories, fetchCategories } = useAdminCategories()
const { tags, fetchTags } = useAdminTags()
const toast = useToast()

const saving = ref(false)
const selectedTags = ref<{ id: number; name: string }[]>([])
const activeTab = ref(0)

const articleForm = ref({
  title: '',
  slug: '',
  excerpt: '',
  cover: '',
  content: '',
  status: 'draft',
  difficulty: 'beginner',
  publishAt: '',
  metaTitle: '',
  metaDesc: '',
  categoryId: null as number | null,
  tagIds: [] as number[]
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

const generateSlug = () => {
  if (articleForm.value.title && !articleForm.value.slug) {
    articleForm.value.slug = articleForm.value.title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
  }
}

const addTag = (tagId: number) => {
  const tag = tags.value.find(t => t.id === tagId)
  if (tag && !selectedTags.value.find(t => t.id === tagId)) {
    selectedTags.value.push({ id: tag.id, name: tag.name })
    articleForm.value.tagIds.push(tagId)
  }
}

const removeTag = (tagId: number) => {
  selectedTags.value = selectedTags.value.filter(t => t.id !== tagId)
  articleForm.value.tagIds = articleForm.value.tagIds.filter(id => id !== tagId)
}

// Play TTS using Web Speech API
const playTts = (text: string) => {
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
  if (!articleForm.value.content) return

  processing.value.split = true
  progress.value = {
    show: true,
    title: 'Splitting Content',
    current: 0,
    total: 1,
    currentItem: 'Processing...'
  }

  try {
    const result = await $fetch('/api/admin/articles/split', {
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
      const paragraphResult = await $fetch('/api/admin/articles/translate', {
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
      const sentenceResult = await $fetch('/api/admin/articles/translate', {
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
      const paragraphResult = await $fetch('/api/admin/articles/tts', {
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
      const sentenceResult = await $fetch('/api/admin/articles/tts', {
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
    const result = await $fetch('/api/admin/articles/phonetics', {
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

const saveDraft = async () => {
  articleForm.value.status = 'draft'
  await saveArticle()
}

const publish = async () => {
  if (articleForm.value.status === 'draft') {
    articleForm.value.status = 'published'
  }
  await saveArticle()
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
      categoryId: articleForm.value.categoryId || undefined,
      tagIds: articleForm.value.tagIds.length > 0 ? articleForm.value.tagIds : undefined
    }

    // Include splitContent if we have it
    if (hasSplitContent.value) {
      data.splitContent = splitContent.value
    }

    const article = await createArticle(data)
    await navigateTo(`/admin/articles/${article.id}/edit`)
  } catch (e) {
    // Error is handled in the composable
  } finally {
    saving.value = false
    pendingSave.value = false
  }
}

onMounted(async () => {
  await Promise.all([
    fetchCategories(),
    fetchTags()
  ])
})
</script>