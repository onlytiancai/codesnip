<template>
  <NuxtLayout name="admin">
    <div class="max-w-4xl">
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
                <UInput v-model="articleForm.title" placeholder="Enter article title" @input="generateSlug" />
              </UFormField>
              <UFormField label="Slug" name="slug" required>
                <UInput v-model="articleForm.slug" placeholder="article-slug" />
              </UFormField>
              <UFormField label="Excerpt" name="excerpt">
                <UTextarea v-model="articleForm.excerpt" placeholder="Brief description of the article" :rows="2" />
              </UFormField>
              <UFormField label="Content" name="content" required>
                <UTextarea
                  v-model="articleForm.content"
                  placeholder="Write or paste your article content here..."
                  :rows="15"
                />
              </UFormField>
            </div>
          </UCard>

          <!-- SEO Settings -->
          <UCard>
            <template #header>
              <h3 class="font-semibold">SEO Settings</h3>
            </template>
            <div class="space-y-4">
              <UFormField label="Meta Title" name="metaTitle">
                <UInput v-model="articleForm.metaTitle" placeholder="SEO title (optional)" />
              </UFormField>
              <UFormField label="Meta Description" name="metaDesc">
                <UTextarea v-model="articleForm.metaDesc" placeholder="SEO description (optional)" :rows="2" />
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
                />
              </UFormField>
              <UFormField label="Publish Date" name="publishAt">
                <UInput v-model="articleForm.publishAt" type="datetime-local" />
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
              <UInput v-model="articleForm.cover" placeholder="https://example.com/image.jpg" />
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
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false,
  middleware: 'admin'
})

const { createArticle, loading } = useAdminArticles()
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
  publishAt: '',
  metaTitle: '',
  metaDesc: '',
  categoryId: null as number | null,
  tagIds: [] as number[]
})

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
  saving.value = true
  try {
    const data = {
      ...articleForm.value,
      categoryId: articleForm.value.categoryId || undefined,
      tagIds: articleForm.value.tagIds.length > 0 ? articleForm.value.tagIds : undefined
    }

    const article = await createArticle(data)
    await navigateTo(`/admin/articles/${article.id}/edit`)
  } catch (e) {
    // Error is handled in the composable
  } finally {
    saving.value = false
  }
}

onMounted(async () => {
  await Promise.all([
    fetchCategories(),
    fetchTags()
  ])
})
</script>