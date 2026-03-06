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
                <UInput placeholder="Enter article title" />
              </UFormField>
              <UFormField label="Excerpt" name="excerpt">
                <UTextarea placeholder="Brief description of the article" :rows="2" />
              </UFormField>
              <UFormField label="Content" name="content" required>
                <UTextarea
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
                <UInput placeholder="SEO title (optional)" />
              </UFormField>
              <UFormField label="Meta Description" name="metaDescription">
                <UTextarea placeholder="SEO description (optional)" :rows="2" />
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
                  :items="[
                    { label: 'Draft', value: 'draft' },
                    { label: 'Published', value: 'published' }
                  ]"
                  default-value="draft"
                />
              </UFormField>
              <UFormField label="Publish Date" name="publishDate">
                <UInput type="date" />
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
                  :items="categories"
                  placeholder="Select category"
                />
              </UFormField>
              <UFormField label="Difficulty" name="difficulty" required>
                <USelect
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
                  <UBadge v-for="tag in selectedTags" :key="tag" color="primary" variant="subtle">
                    {{ tag }}
                    <button @click="removeTag(tag)" class="ml-1">
                      <UIcon name="i-lucide-x" class="w-3 h-3" />
                    </button>
                  </UBadge>
                </div>
                <div class="flex gap-2">
                  <UInput v-model="newTag" placeholder="Add tag" class="flex-1" />
                  <UButton @click="addTag" variant="outline">Add</UButton>
                </div>
              </UFormField>
            </div>
          </UCard>

          <!-- Cover Image -->
          <UCard>
            <template #header>
              <h3 class="font-semibold">Cover Image</h3>
            </template>
            <div class="border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg p-6 text-center">
              <UIcon name="i-lucide-upload" class="w-8 h-8 text-gray-400 mx-auto mb-2" />
              <p class="text-sm text-gray-500 dark:text-gray-400 mb-2">
                Drag and drop or click to upload
              </p>
              <p class="text-xs text-gray-400">PNG, JPG up to 5MB</p>
              <UButton size="sm" variant="outline" class="mt-3">
                Choose File
              </UButton>
            </div>
          </UCard>

          <!-- Actions -->
          <div class="flex gap-3">
            <UButton variant="outline" class="flex-1">Save Draft</UButton>
            <UButton color="primary" class="flex-1">Publish</UButton>
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

const newTag = ref('')
const selectedTags = ref(['technology', 'AI'])

const categories = [
  { label: 'Technology', value: 'technology' },
  { label: 'Science', value: 'science' },
  { label: 'Business', value: 'business' },
  { label: 'Health', value: 'health' },
  { label: 'Culture', value: 'culture' },
  { label: 'Travel', value: 'travel' }
]

const addTag = () => {
  if (newTag.value && !selectedTags.value.includes(newTag.value)) {
    selectedTags.value.push(newTag.value)
    newTag.value = ''
  }
}

const removeTag = (tag: string) => {
  selectedTags.value = selectedTags.value.filter(t => t !== tag)
}
</script>