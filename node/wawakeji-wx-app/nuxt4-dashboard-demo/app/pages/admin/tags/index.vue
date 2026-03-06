<template>
  <NuxtLayout name="admin">
    <div>
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-2xl font-bold">Tags</h2>
        <UButton icon="i-lucide-plus" @click="showModal = true">
          Add Tag
        </UButton>
      </div>

      <!-- Stats -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        <UCard class="text-center">
          <p class="text-2xl font-bold">{{ tags.length }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Total Tags</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold">{{ tags.reduce((sum, t) => sum + t.articleCount, 0) }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Tagged Articles</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold">{{ tags.filter(t => t.articleCount > 10).length }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Popular Tags</p>
        </UCard>
      </div>

      <!-- Search & Filter -->
      <div class="flex items-center gap-3 mb-6">
        <UInput
          placeholder="Search tags..."
          icon="i-lucide-search"
          class="w-64"
        />
        <USelect
          :items="[
            { label: 'All Tags', value: 'all' },
            { label: 'Popular (10+)', value: 'popular' },
            { label: 'Unused', value: 'unused' }
          ]"
          default-value="all"
          class="w-40"
        />
      </div>

      <!-- Tags Grid -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        <UCard
          v-for="tag in tags"
          :key="tag.id"
          class="hover:border-primary transition cursor-pointer"
        >
          <div class="flex items-start justify-between">
            <div class="flex items-center gap-3">
              <UBadge :style="{ backgroundColor: tag.color }" class="text-white">
                {{ tag.name }}
              </UBadge>
              <span class="text-sm text-gray-500">{{ tag.articleCount }} articles</span>
            </div>
            <UDropdownMenu :items="getActionItems(tag)">
              <UButton icon="i-lucide-more-horizontal" color="neutral" variant="ghost" size="xs" />
            </UDropdownMenu>
          </div>
          <p v-if="tag.description" class="text-sm text-gray-500 dark:text-gray-400 mt-2">
            {{ tag.description }}
          </p>
          <div class="flex items-center gap-2 mt-3 text-xs text-gray-400">
            <span>Created {{ tag.createdAt }}</span>
          </div>
        </UCard>
      </div>

      <!-- Add Tag Modal -->
      <UModal v-model:open="showModal">
        <UCard>
          <template #header>
            <h3 class="text-lg font-semibold">Add Tag</h3>
          </template>
          <div class="space-y-4">
            <UFormField label="Tag Name" name="name" required>
              <UInput placeholder="Tag name" />
            </UFormField>
            <UFormField label="Slug" name="slug">
              <UInput placeholder="tag-slug" />
            </UFormField>
            <UFormField label="Description" name="description">
              <UTextarea placeholder="Tag description (optional)" :rows="2" />
            </UFormField>
            <UFormField label="Color" name="color">
              <div class="flex gap-2 flex-wrap">
                <div
                  v-for="color in colors"
                  :key="color"
                  :class="[
                    'w-8 h-8 rounded-lg cursor-pointer transition ring-2',
                    selectedColor === color ? 'ring-primary ring-offset-2' : 'ring-transparent'
                  ]"
                  :style="{ backgroundColor: color }"
                  @click="selectedColor = color"
                />
              </div>
            </UFormField>
          </div>
          <template #footer>
            <div class="flex justify-end gap-3">
              <UButton variant="outline" @click="showModal = false">Cancel</UButton>
              <UButton color="primary">Create Tag</UButton>
            </div>
          </template>
        </UCard>
      </UModal>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false
})

const showModal = ref(false)
const selectedColor = ref('#3b82f6')

const colors = [
  '#3b82f6', '#22c55e', '#a855f7', '#ef4444', '#f97316', '#06b6d4',
  '#ec4899', '#8b5cf6', '#14b8a6', '#f59e0b', '#6366f1', '#84cc16'
]

const tags = [
  { id: 1, name: 'AI', slug: 'ai', color: '#3b82f6', articleCount: 28, description: 'Artificial Intelligence related', createdAt: 'Jan 15, 2026' },
  { id: 2, name: 'Healthcare', slug: 'healthcare', color: '#ef4444', articleCount: 15, description: '', createdAt: 'Jan 20, 2026' },
  { id: 3, name: 'Climate', slug: 'climate', color: '#22c55e', articleCount: 12, description: 'Climate change and environment', createdAt: 'Feb 1, 2026' },
  { id: 4, name: 'Startup', slug: 'startup', color: '#a855f7', articleCount: 23, description: '', createdAt: 'Feb 5, 2026' },
  { id: 5, name: 'Mental Health', slug: 'mental-health', color: '#f97316', articleCount: 8, description: '', createdAt: 'Feb 10, 2026' },
  { id: 6, name: 'Finance', slug: 'finance', color: '#06b6d4', articleCount: 19, description: '', createdAt: 'Feb 15, 2026' },
  { id: 7, name: 'Machine Learning', slug: 'machine-learning', color: '#8b5cf6', articleCount: 14, description: '', createdAt: 'Feb 20, 2026' },
  { id: 8, name: 'Sleep', slug: 'sleep', color: '#ec4899', articleCount: 6, description: '', createdAt: 'Feb 25, 2026' }
]

const getActionItems = (tag: any) => [
  [{
    label: 'Edit',
    icon: 'i-lucide-edit'
  }],
  [{
    label: 'Delete',
    icon: 'i-lucide-trash-2',
    color: 'error'
  }]
]
</script>