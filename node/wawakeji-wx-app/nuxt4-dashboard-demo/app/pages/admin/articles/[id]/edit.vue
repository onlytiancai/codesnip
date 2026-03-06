<template>
  <NuxtLayout name="admin">
    <div class="max-w-4xl">
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <div class="flex items-center gap-3">
          <UButton variant="ghost" to="/admin/articles">
            <UIcon name="i-lucide-arrow-left" class="w-4 h-4 mr-2" />
            Back
          </UButton>
          <h2 class="text-2xl font-bold">Edit Article</h2>
        </div>
        <UBadge :color="article.status === 'published' ? 'success' : 'warning'">
          {{ article.status }}
        </UBadge>
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
                <UInput v-model="article.title" placeholder="Enter article title" />
              </UFormField>
              <UFormField label="Excerpt" name="excerpt">
                <UTextarea v-model="article.excerpt" placeholder="Brief description" :rows="2" />
              </UFormField>
              <UFormField label="Content" name="content" required>
                <UTextarea
                  v-model="article.content"
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
                <UBadge color="primary" variant="subtle">Premium</UBadge>
              </div>
            </template>
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <UButton variant="outline" block icon="i-lucide-scissors">
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
                <h3 class="font-semibold">Sentences ({{ sentences.length }})</h3>
                <UButton size="sm" variant="outline" icon="i-lucide-plus">
                  Add Sentence
                </UButton>
              </div>
            </template>
            <div class="space-y-4">
              <div
                v-for="(sentence, index) in sentences"
                :key="index"
                class="p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
              >
                <div class="flex items-start gap-3">
                  <span class="text-sm text-gray-400 w-6">{{ index + 1 }}.</span>
                  <div class="flex-1 space-y-2">
                    <UInput v-model="sentence.en" placeholder="English sentence" />
                    <UInput v-model="sentence.cn" placeholder="Chinese translation" />
                    <div class="flex items-center gap-2">
                      <UButton size="xs" variant="ghost" icon="i-lucide-volume-2" />
                      <UButton size="xs" variant="ghost" icon="i-lucide-edit" />
                      <UButton size="xs" variant="ghost" color="error" icon="i-lucide-trash-2" />
                    </div>
                  </div>
                </div>
              </div>
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
                  v-model="article.status"
                  :items="[
                    { label: 'Draft', value: 'draft' },
                    { label: 'Published', value: 'published' }
                  ]"
                />
              </UFormField>
              <UFormField label="Last Updated" name="updatedAt">
                <p class="text-sm text-gray-500">{{ article.updatedAt }}</p>
              </UFormField>
            </div>
          </UCard>

          <!-- Classification -->
          <UCard>
            <template #header>
              <h3 class="font-semibold">Classification</h3>
            </template>
            <div class="space-y-4">
              <UFormField label="Category" name="category" required>
                <USelect
                  v-model="article.category"
                  :items="categories"
                />
              </UFormField>
              <UFormField label="Difficulty" name="difficulty" required>
                <USelect
                  v-model="article.difficulty"
                  :items="[
                    { label: 'Beginner', value: 'beginner' },
                    { label: 'Intermediate', value: 'intermediate' },
                    { label: 'Advanced', value: 'advanced' }
                  ]"
                />
              </UFormField>
              <UFormField label="Tags" name="tags">
                <div class="flex flex-wrap gap-2">
                  <UBadge v-for="tag in article.tags" :key="tag" color="primary" variant="subtle">
                    {{ tag }}
                    <button class="ml-1">
                      <UIcon name="i-lucide-x" class="w-3 h-3" />
                    </button>
                  </UBadge>
                </div>
              </UFormField>
            </div>
          </UCard>

          <!-- Cover Image -->
          <UCard>
            <template #header>
              <h3 class="font-semibold">Cover Image</h3>
            </template>
            <img
              :src="article.cover"
              :alt="article.title"
              class="w-full h-40 object-cover rounded-lg mb-3"
            />
            <UButton size="sm" variant="outline" block>
              Change Image
            </UButton>
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
            <UButton variant="outline" class="flex-1">Preview</UButton>
            <UButton color="primary" class="flex-1">Save Changes</UButton>
          </div>
        </div>
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false
})

const route = useRoute()

const article = ref({
  id: route.params.id,
  title: 'The Future of Artificial Intelligence in Healthcare',
  excerpt: 'Explore how AI is revolutionizing medical diagnosis and treatment.',
  content: 'Artificial intelligence is transforming the healthcare industry...',
  status: 'published',
  category: 'technology',
  difficulty: 'intermediate',
  tags: ['AI', 'Healthcare', 'Technology'],
  cover: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=400&h=300&fit=crop',
  views: '2.3k',
  bookmarks: 156,
  updatedAt: 'Mar 5, 2026 at 2:30 PM'
})

const sentences = ref([
  { en: 'Artificial intelligence is transforming the healthcare industry.', cn: '人工智能正在改变医疗行业。' },
  { en: 'From diagnostic tools to personalized treatment plans, AI is revolutionizing patient care.', cn: '从诊断工具到个性化治疗方案，AI正在彻底改变患者护理。' },
  { en: 'One of the most promising applications is in medical imaging analysis.', cn: '最有前途的应用之一是医学影像分析。' }
])

const categories = [
  { label: 'Technology', value: 'technology' },
  { label: 'Science', value: 'science' },
  { label: 'Business', value: 'business' },
  { label: 'Health', value: 'health' },
  { label: 'Culture', value: 'culture' }
]
</script>