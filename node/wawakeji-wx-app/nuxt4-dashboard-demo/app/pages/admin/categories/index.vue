<template>
  <NuxtLayout name="admin">
    <div>
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-2xl font-bold">Categories</h2>
        <UButton icon="i-lucide-plus" @click="showModal = true">
          Add Category
        </UButton>
      </div>

      <!-- Categories Table -->
      <UCard>
        <UTable :data="categories" :columns="columns">
          <template #name-cell="{ row }">
            <div class="flex items-center gap-3">
              <div :class="row.original.iconBg" class="w-10 h-10 rounded-lg flex items-center justify-center">
                <UIcon :name="row.original.icon" class="w-5 h-5 text-white" />
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
              <UButton icon="i-lucide-edit" color="neutral" variant="ghost" size="xs" @click="editCategory(row.original)" />
              <UButton icon="i-lucide-trash-2" color="error" variant="ghost" size="xs" />
            </div>
          </template>
        </UTable>
      </UCard>

      <!-- Add/Edit Modal -->
      <UModal v-model:open="showModal">
        <UCard>
          <template #header>
            <h3 class="text-lg font-semibold">{{ editingCategory ? 'Edit Category' : 'Add Category' }}</h3>
          </template>
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
          <template #footer>
            <div class="flex justify-end gap-3">
              <UButton variant="outline" @click="showModal = false">Cancel</UButton>
              <UButton color="primary">{{ editingCategory ? 'Update' : 'Create' }}</UButton>
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
const editingCategory = ref<any>(null)

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
  { id: 'articleCount', header: 'Articles' },
  { id: 'status', header: 'Status' },
  { id: 'actions', header: '' }
]

const categories = [
  { id: 1, name: 'Technology', slug: 'technology', icon: 'i-lucide-cpu', iconBg: 'bg-blue-500', articleCount: 42, status: 'active' },
  { id: 2, name: 'Science', slug: 'science', icon: 'i-lucide-flask-conical', iconBg: 'bg-green-500', articleCount: 38, status: 'active' },
  { id: 3, name: 'Business', slug: 'business', icon: 'i-lucide-briefcase', iconBg: 'bg-purple-500', articleCount: 56, status: 'active' },
  { id: 4, name: 'Health', slug: 'health', icon: 'i-lucide-heart-pulse', iconBg: 'bg-red-500', articleCount: 31, status: 'active' },
  { id: 5, name: 'Culture', slug: 'culture', icon: 'i-lucide-globe', iconBg: 'bg-orange-500', articleCount: 27, status: 'active' },
  { id: 6, name: 'Travel', slug: 'travel', icon: 'i-lucide-plane', iconBg: 'bg-cyan-500', articleCount: 19, status: 'inactive' }
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

const editCategory = (category: any) => {
  editingCategory.value = category
  categoryForm.value = {
    name: category.name,
    slug: category.slug,
    description: '',
    icon: category.icon,
    color: '#3b82f6',
    status: category.status
  }
  showModal.value = true
}
</script>