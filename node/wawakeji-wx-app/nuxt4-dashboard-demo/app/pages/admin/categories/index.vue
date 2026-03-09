<template>
  <NuxtLayout name="admin">
    <div>
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-2xl font-bold">Categories</h2>
        <UButton icon="i-lucide-plus" @click="openCreateModal">
          Add Category
        </UButton>
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="flex items-center justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
      </div>

      <!-- Categories Table -->
      <UCard v-else>
        <UTable :data="categories" :columns="columns">
          <template #name-cell="{ row }">
            <div class="flex items-center gap-3">
              <div
                :style="{ backgroundColor: row.original.color }"
                class="w-10 h-10 rounded-lg flex items-center justify-center"
              >
                <UIcon :name="row.original.icon || 'i-lucide-folder'" class="w-5 h-5 text-white" />
              </div>
              <div>
                <p class="font-medium">{{ row.original.name }}</p>
                <p class="text-xs text-gray-500 dark:text-gray-400">{{ row.original.slug }}</p>
              </div>
            </div>
          </template>
          <template #status-cell="{ row }">
            <UBadge :color="row.original.status === 'active' ? 'success' : 'neutral'" variant="subtle" size="xs">
              {{ row.original.status }}
            </UBadge>
          </template>
          <template #actions-cell="{ row }">
            <div class="flex items-center gap-1">
              <UButton
                icon="i-lucide-edit"
                color="neutral"
                variant="ghost"
                size="xs"
                @click="openEditModal(row.original)"
              />
              <UButton
                icon="i-lucide-trash-2"
                color="error"
                variant="ghost"
                size="xs"
                @click="confirmDelete(row.original)"
              />
            </div>
          </template>
        </UTable>
      </UCard>

      <!-- Add/Edit Modal -->
      <UModal
        v-model:open="showModal"
        :title="editingCategory ? 'Edit Category' : 'Add Category'"
        description="Configure the category details below"
      >
        <template #body>
          <div class="space-y-4">
            <UFormField label="Name" name="name" required>
              <UInput v-model="categoryForm.name" placeholder="Category name" />
            </UFormField>
            <UFormField label="Slug" name="slug" required>
              <UInput v-model="categoryForm.slug" placeholder="category-slug" />
            </UFormField>
            <UFormField label="Description" name="description">
              <UTextarea v-model="categoryForm.description" placeholder="Category description" :rows="2" />
            </UFormField>
            <UFormField label="Icon" name="icon">
              <div class="grid grid-cols-6 gap-2">
                <div
                  v-for="icon in icons"
                  :key="icon"
                  :class="[
                    'p-3 rounded-lg border cursor-pointer transition',
                    categoryForm.icon === icon
                      ? 'border-primary bg-primary/5'
                      : 'border-gray-200 dark:border-gray-700 hover:border-primary/50'
                  ]"
                  @click="categoryForm.icon = icon"
                >
                  <UIcon :name="icon" class="w-5 h-5" />
                </div>
              </div>
            </UFormField>
            <UFormField label="Color" name="color">
              <div class="flex gap-2">
                <div
                  v-for="color in colors"
                  :key="color"
                  :class="[
                    'w-8 h-8 rounded-lg cursor-pointer transition ring-2',
                    categoryForm.color === color ? 'ring-primary ring-offset-2' : 'ring-transparent'
                  ]"
                  :style="{ backgroundColor: color }"
                  @click="categoryForm.color = color"
                />
              </div>
            </UFormField>
            <UFormField label="Status" name="status">
              <USelect
                v-model="categoryForm.status"
                :items="[
                  { label: 'Active', value: 'active' },
                  { label: 'Inactive', value: 'inactive' }
                ]"
              />
            </UFormField>
          </div>
        </template>
        <template #footer>
          <div class="flex justify-end gap-3">
            <UButton variant="outline" @click="showModal = false">Cancel</UButton>
            <UButton color="primary" :loading="saving" @click="handleSave">
              {{ editingCategory ? 'Update' : 'Create' }}
            </UButton>
          </div>
        </template>
      </UModal>

      <!-- Delete Confirmation -->
      <UModal v-model:open="showDeleteModal" title="Delete Category" description="Are you sure you want to delete this category?">
        <template #body>
          <p class="text-gray-500">
            This action cannot be undone. The category "{{ categoryToDelete?.name }}" will be permanently deleted.
            <span v-if="categoryToDelete?.articleCount" class="text-red-500 block mt-2">
              Warning: This category has {{ categoryToDelete.articleCount }} articles. Please reassign them first.
            </span>
          </p>
        </template>
        <template #footer>
          <div class="flex justify-end gap-3">
            <UButton variant="outline" @click="showDeleteModal = false">Cancel</UButton>
            <UButton
              color="error"
              :loading="deleting"
              :disabled="!!categoryToDelete?.articleCount"
              @click="handleDelete"
            >
              Delete
            </UButton>
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
  categories,
  loading,
  fetchCategories,
  createCategory,
  updateCategory,
  deleteCategory
} = useAdminCategories()

const showModal = ref(false)
const showDeleteModal = ref(false)
const editingCategory = ref<any>(null)
const categoryToDelete = ref<any>(null)
const saving = ref(false)
const deleting = ref(false)

const categoryForm = ref({
  name: '',
  slug: '',
  description: '',
  icon: 'i-lucide-folder',
  color: '#3b82f6',
  status: 'active'
})

const columns = [
  { id: 'name', header: 'Category' },
  { id: 'articleCount', header: 'Articles', accessorKey: 'articleCount' },
  { id: 'status', header: 'Status' },
  { id: 'actions', header: '' }
]

const icons = [
  'i-lucide-cpu',
  'i-lucide-flask-conical',
  'i-lucide-briefcase',
  'i-lucide-heart-pulse',
  'i-lucide-globe',
  'i-lucide-plane',
  'i-lucide-trophy',
  'i-lucide-film',
  'i-lucide-graduation-cap',
  'i-lucide-music',
  'i-lucide-utensils',
  'i-lucide-book-open'
]

const colors = ['#3b82f6', '#22c55e', '#a855f7', '#ef4444', '#f97316', '#06b6d4']

const openCreateModal = () => {
  editingCategory.value = null
  categoryForm.value = {
    name: '',
    slug: '',
    description: '',
    icon: 'i-lucide-folder',
    color: '#3b82f6',
    status: 'active'
  }
  showModal.value = true
}

const openEditModal = (category: any) => {
  editingCategory.value = category
  categoryForm.value = {
    name: category.name,
    slug: category.slug,
    description: category.description || '',
    icon: category.icon || 'i-lucide-folder',
    color: category.color || '#3b82f6',
    status: category.status
  }
  showModal.value = true
}

const handleSave = async () => {
  saving.value = true
  try {
    if (editingCategory.value) {
      await updateCategory(editingCategory.value.id, categoryForm.value)
    } else {
      await createCategory(categoryForm.value)
    }
    showModal.value = false
  } catch (e) {
    // Error is handled in the composable
  } finally {
    saving.value = false
  }
}

const confirmDelete = (category: any) => {
  categoryToDelete.value = category
  showDeleteModal.value = true
}

const handleDelete = async () => {
  if (!categoryToDelete.value) return

  deleting.value = true
  try {
    await deleteCategory(categoryToDelete.value.id)
    showDeleteModal.value = false
    categoryToDelete.value = null
  } catch (e) {
    // Error is handled in the composable
  } finally {
    deleting.value = false
  }
}

// Auto-generate slug from name
watch(() => categoryForm.value.name, (name) => {
  if (!editingCategory.value) {
    categoryForm.value.slug = name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
  }
})

onMounted(() => {
  fetchCategories()
})
</script>