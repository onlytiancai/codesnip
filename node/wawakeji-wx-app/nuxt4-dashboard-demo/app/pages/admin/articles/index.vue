<template>
  <NuxtLayout name="admin">
    <div>
      <!-- Header -->
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div class="flex items-center gap-3">
          <UInput
            v-model="searchQuery"
            placeholder="Search articles..."
            icon="i-lucide-search"
            class="w-64"
            @input="debouncedSearch"
          />
          <USelect
            v-model="statusFilter"
            :items="statusOptions"
            placeholder="Status"
            class="w-32"
            @change="applyFilters"
          />
          <USelect
            v-model="categoryFilter"
            :items="categoryOptions"
            placeholder="Category"
            class="w-40"
            @change="applyFilters"
          />
        </div>
        <UButton to="/admin/articles/create" icon="i-lucide-plus">
          New Article
        </UButton>
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="flex items-center justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
      </div>

      <!-- Articles Table -->
      <UCard v-else>
        <UTable :data="articles" :columns="columns">
          <template #title-cell="{ row }">
            <div class="flex items-center gap-3">
              <img
                v-if="row.original.cover"
                :src="row.original.cover"
                :alt="row.original.title"
                class="w-12 h-12 object-cover rounded"
              />
              <div class="w-12 h-12 bg-gray-100 dark:bg-gray-800 rounded flex items-center justify-center" v-else>
                <UIcon name="i-lucide-file-text" class="w-6 h-6 text-gray-400" />
              </div>
              <div>
                <p class="font-medium">{{ row.original.title }}</p>
                <p class="text-xs text-gray-500 dark:text-gray-400 line-clamp-1">
                  {{ row.original.excerpt || 'No excerpt' }}
                </p>
              </div>
            </div>
          </template>
          <template #category-cell="{ row }">
            <UBadge v-if="row.original.category" color="primary" variant="subtle" size="xs">
              {{ row.original.category.name }}
            </UBadge>
            <span v-else class="text-gray-400">-</span>
          </template>
          <template #difficulty-cell="{ row }">
            <UBadge :color="difficultyColor(row.original.difficulty)" variant="subtle" size="xs">
              {{ row.original.difficulty }}
            </UBadge>
          </template>
          <template #status-cell="{ row }">
            <UBadge :color="row.original.status === 'published' ? 'success' : 'warning'" variant="subtle" size="xs">
              {{ row.original.status }}
            </UBadge>
          </template>
          <template #actions-cell="{ row }">
            <UDropdownMenu :items="getActionItems(row.original)">
              <UButton icon="i-lucide-more-horizontal" color="neutral" variant="ghost" size="xs" />
            </UDropdownMenu>
          </template>
        </UTable>

        <!-- Pagination -->
        <div class="flex items-center justify-between mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <p class="text-sm text-gray-500 dark:text-gray-400">
            Showing {{ (pagination.page - 1) * pagination.limit + 1 }}-{{ Math.min(pagination.page * pagination.limit, pagination.total) }} of {{ pagination.total }} articles
          </p>
          <UPagination
            v-model:page="currentPage"
            :total="pagination.total"
            :items-per-page="pagination.limit"
            @update:page="handlePageChange"
          />
        </div>
      </UCard>

      <!-- Delete Confirmation -->
      <UModal v-model:open="showDeleteModal" title="Delete Article" description="Are you sure you want to delete this article?">
        <template #body>
          <p class="text-gray-500">
            This action cannot be undone. The article "{{ articleToDelete?.title }}" will be permanently deleted.
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
  articles,
  pagination,
  loading,
  fetchArticles,
  deleteArticle
} = useAdminArticles()

const { categories, fetchCategories } = useAdminCategories()

const currentPage = ref(1)
const searchQuery = ref('')
const statusFilter = ref('all')
const categoryFilter = ref('all')
const showDeleteModal = ref(false)
const articleToDelete = ref<any>(null)
const deleting = ref(false)

const statusOptions = [
  { label: 'All Status', value: 'all' },
  { label: 'Published', value: 'published' },
  { label: 'Draft', value: 'draft' }
]

const categoryOptions = computed(() => [
  { label: 'All Categories', value: 'all' },
  ...categories.value.map(c => ({ label: c.name, value: c.id.toString() }))
])

const columns = [
  { id: 'title', header: 'Article' },
  { id: 'category', header: 'Category' },
  { id: 'difficulty', header: 'Difficulty' },
  { id: 'views', header: 'Views', accessorKey: 'views' },
  { id: 'status', header: 'Status' },
  { id: 'createdAt', header: 'Created', accessorKey: 'createdAt' },
  { id: 'actions', header: '' }
]

const difficultyColor = (difficulty: string) => {
  switch (difficulty) {
    case 'beginner': return 'success'
    case 'intermediate': return 'warning'
    case 'advanced': return 'error'
    default: return 'neutral'
  }
}

const applyFilters = () => {
  currentPage.value = 1
  fetchData()
}

const handlePageChange = (page: number) => {
  currentPage.value = page
  fetchData()
}

const fetchData = () => {
  fetchArticles({
    page: currentPage.value,
    status: statusFilter.value !== 'all' ? statusFilter.value : undefined,
    categoryId: categoryFilter.value !== 'all' ? categoryFilter.value : undefined,
    search: searchQuery.value || undefined
  })
}

// Debounced search
let searchTimeout: NodeJS.Timeout
const debouncedSearch = () => {
  clearTimeout(searchTimeout)
  searchTimeout = setTimeout(() => {
    currentPage.value = 1
    fetchData()
  }, 300)
}

const confirmDelete = (article: any) => {
  articleToDelete.value = article
  showDeleteModal.value = true
}

const handleDelete = async () => {
  if (!articleToDelete.value) return

  deleting.value = true
  try {
    await deleteArticle(articleToDelete.value.id)
    showDeleteModal.value = false
    articleToDelete.value = null
  } catch (e) {
    // Error is handled in the composable
  } finally {
    deleting.value = false
  }
}

const getActionItems = (article: any) => [
  [{
    label: 'Edit',
    icon: 'i-lucide-edit',
    to: `/admin/articles/${article.id}/edit`
  }, {
    label: 'Preview',
    icon: 'i-lucide-eye',
    to: `/articles/${article.slug}`
  }],
  [{
    label: 'Delete',
    icon: 'i-lucide-trash-2',
    color: 'error' as const,
    click: () => confirmDelete(article)
  }]
]

onMounted(async () => {
  await fetchCategories()
  await fetchData()
})
</script>