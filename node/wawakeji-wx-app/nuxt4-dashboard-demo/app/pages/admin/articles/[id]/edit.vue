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
                  <UTextarea
                    v-model="articleForm.content"
                    placeholder="Write or paste your article content..."
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
                  <UBadge color="primary" variant="subtle">Premium</UBadge>
                </div>
              </template>
              <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <UButton variant="outline" block icon="i-lucide-scissors" @click="splitSentences">
                  Split Sentences
                </UButton>
                <UButton variant="outline" block icon="i-lucide-languages">
                  Translate
                </UButton>
                <UButton variant="outline" block icon="i-lucide-volume-2">
                  Generate TTS
                </UButton>
              </div>
            </UCard>

            <!-- Sentences Editor -->
            <UCard>
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
                  No sentences yet. Click "Add Sentence" or use "Split Sentences" to generate from content.
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
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false,
  middleware: 'admin'
})

const route = useRoute()
const articleId = parseInt(route.params.id as string)

const { article, loading, fetchArticle, updateArticle } = useAdminArticles()
const { categories, fetchCategories } = useAdminCategories()
const { tags, fetchTags } = useAdminTags()

const saving = ref(false)
const selectedTags = ref<{ id: number; name: string }[]>([])

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
  // Re-order
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

  // Re-order
  articleForm.value.sentences.forEach((s, i) => {
    s.order = i
  })
}

const splitSentences = () => {
  if (!articleForm.value.content) return

  // Simple sentence splitting by period
  const sentences = articleForm.value.content
    .split(/[.!?]+/)
    .map(s => s.trim())
    .filter(s => s.length > 10)
    .map((s, i) => ({
      order: i,
      en: s + '.',
      cn: ''
    }))

  articleForm.value.sentences = sentences
}

const saveArticle = async () => {
  saving.value = true
  try {
    const data = {
      ...articleForm.value,
      categoryId: articleForm.value.categoryId || null,
      tagIds: selectedTags.value.map(t => t.id),
      sentences: articleForm.value.sentences.map((s, i) => ({
        ...s,
        order: i
      }))
    }

    await updateArticle(articleId, data)
  } catch (e) {
    // Error is handled in the composable
  } finally {
    saving.value = false
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