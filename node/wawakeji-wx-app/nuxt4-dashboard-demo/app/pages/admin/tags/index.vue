<template>
  <NuxtLayout name="admin">
    <div>
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-2xl font-bold">Tags</h2>
        <UButton icon="i-lucide-plus" @click="openCreateModal">
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
          <p class="text-2xl font-bold">{{ totalTaggedArticles }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Tagged Articles</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold">{{ popularTagsCount }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Popular Tags (10+)</p>
        </UCard>
      </div>

      <!-- Search -->
      <div class="flex items-center gap-3 mb-6">
        <UInput
          v-model="searchQuery"
          placeholder="Search tags..."
          icon="i-lucide-search"
          class="w-64"
          @input="debouncedSearch"
        />
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="flex items-center justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
      </div>

      <!-- Tags Grid -->
      <div v-else class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        <UCard
          v-for="tag in filteredTags"
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
            <span>Created {{ formatDate(tag.createdAt) }}</span>
          </div>
        </UCard>
      </div>

      <!-- Add/Edit Tag Modal -->
      <UModal
        v-model:open="showModal"
        :title="editingTag ? 'Edit Tag' : 'Add Tag'"
        description="Create a new tag for organizing articles"
      >
        <template #body>
          <div class="space-y-4">
            <UFormField label="Tag Name" name="name" required>
              <UInput v-model="tagForm.name" placeholder="Tag name" />
            </UFormField>
            <UFormField label="Slug" name="slug">
              <UInput v-model="tagForm.slug" placeholder="tag-slug" />
            </UFormField>
            <UFormField label="Description" name="description">
              <UTextarea v-model="tagForm.description" placeholder="Tag description (optional)" :rows="2" />
            </UFormField>
            <UFormField label="Color" name="color">
              <div class="flex gap-2 flex-wrap">
                <div
                  v-for="color in colors"
                  :key="color"
                  :class="[
                    'w-8 h-8 rounded-lg cursor-pointer transition ring-2',
                    tagForm.color === color ? 'ring-primary ring-offset-2' : 'ring-transparent'
                  ]"
                  :style="{ backgroundColor: color }"
                  @click="tagForm.color = color"
                />
              </div>
            </UFormField>
          </div>
        </template>
        <template #footer>
          <div class="flex justify-end gap-3">
            <UButton variant="outline" @click="showModal = false">Cancel</UButton>
            <UButton color="primary" :loading="saving" @click="handleSave">
              {{ editingTag ? 'Update' : 'Create' }}
            </UButton>
          </div>
        </template>
      </UModal>

      <!-- Delete Confirmation -->
      <UModal v-model:open="showDeleteModal" title="Delete Tag" description="Are you sure you want to delete this tag?">
        <template #body>
          <p class="text-gray-500">
            This action cannot be undone. The tag "{{ tagToDelete?.name }}" will be permanently deleted.
          </p>
        </template>
        <template #footer>
          <div class="flex justify-end gap-3">
            <UButton variant="outline" @click="showDeleteModal = false">Cancel</UButton>
            <UButton color="error" :loading="deleting" @click="handleDelete">Delete</UButton>
          </div>
        </template>
      </UModal>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false,
  middleware: 'admin'
})

const {
  tags,
  loading,
  fetchTags,
  createTag,
  updateTag,
  deleteTag
} = useAdminTags()

const showModal = ref(false)
const showDeleteModal = ref(false)
const editingTag = ref<any>(null)
const tagToDelete = ref<any>(null)
const saving = ref(false)
const deleting = ref(false)
const searchQuery = ref('')

const tagForm = ref({
  name: '',
  slug: '',
  description: '',
  color: '#3b82f6'
})

const colors = [
  '#3b82f6', '#22c55e', '#a855f7', '#ef4444', '#f97316', '#06b6d4',
  '#ec4899', '#8b5cf6', '#14b8a6', '#f59e0b', '#6366f1', '#84cc16'
]

const totalTaggedArticles = computed(() => {
  return tags.value.reduce((sum, t) => sum + t.articleCount, 0)
})

const popularTagsCount = computed(() => {
  return tags.value.filter(t => t.articleCount >= 10).length
})

const filteredTags = computed(() => {
  if (!searchQuery.value) return tags.value
  const query = searchQuery.value.toLowerCase()
  return tags.value.filter(t =>
    t.name.toLowerCase().includes(query) ||
    t.description?.toLowerCase().includes(query)
  )
})

const formatDate = (dateStr: string) => {
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  })
}

const openCreateModal = () => {
  editingTag.value = null
  tagForm.value = {
    name: '',
    slug: '',
    description: '',
    color: '#3b82f6'
  }
  showModal.value = true
}

const openEditModal = (tag: any) => {
  editingTag.value = tag
  tagForm.value = {
    name: tag.name,
    slug: tag.slug,
    description: tag.description || '',
    color: tag.color || '#3b82f6'
  }
  showModal.value = true
}

const handleSave = async () => {
  saving.value = true
  try {
    if (editingTag.value) {
      await updateTag(editingTag.value.id, tagForm.value)
    } else {
      await createTag(tagForm.value)
    }
    showModal.value = false
  } catch (e) {
    // Error is handled in the composable
  } finally {
    saving.value = false
  }
}

const confirmDelete = (tag: any) => {
  tagToDelete.value = tag
  showDeleteModal.value = true
}

const handleDelete = async () => {
  if (!tagToDelete.value) return

  deleting.value = true
  try {
    await deleteTag(tagToDelete.value.id)
    showDeleteModal.value = false
    tagToDelete.value = null
  } catch (e) {
    // Error is handled in the composable
  } finally {
    deleting.value = false
  }
}

const getActionItems = (tag: any) => [
  [{
    label: 'Edit',
    icon: 'i-lucide-edit',
    click: () => openEditModal(tag)
  }],
  [{
    label: 'Delete',
    icon: 'i-lucide-trash-2',
    color: 'error' as const,
    click: () => confirmDelete(tag)
  }]
]

// Auto-generate slug from name
watch(() => tagForm.value.name, (name) => {
  if (!editingTag.value) {
    tagForm.value.slug = name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
  }
})

// Debounced search
let searchTimeout: NodeJS.Timeout
const debouncedSearch = () => {
  clearTimeout(searchTimeout)
  searchTimeout = setTimeout(() => {
    fetchTags({ search: searchQuery.value })
  }, 300)
}

onMounted(() => {
  fetchTags()
})
</script>