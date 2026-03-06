<template>
  <NuxtLayout name="admin">
    <div>
      <!-- Header -->
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div class="flex items-center gap-3">
          <UInput
            placeholder="Search articles..."
            icon="i-lucide-search"
            class="w-64"
          />
          <USelect
            :items="statusOptions"
            placeholder="Status"
            class="w-32"
          />
          <USelect
            :items="categoryOptions"
            placeholder="Category"
            class="w-40"
          />
        </div>
        <UButton to="/admin/articles/create" icon="i-lucide-plus">
          New Article
        </UButton>
      </div>

      <!-- Articles Table -->
      <UCard>
        <UTable :data="articles" :columns="columns">
          <template #title-cell="{ row }">
            <div class="flex items-center gap-3">
              <img
                :src="row.original.cover"
                :alt="row.original.title"
                class="w-12 h-12 object-cover rounded"
              />
              <div>
                <p class="font-medium">{{ row.original.title }}</p>
                <p class="text-xs text-gray-500 dark:text-gray-400">
                  {{ row.original.excerpt }}
                </p>
              </div>
            </div>
          </template>
          <template #category-cell="{ row }">
            <UBadge color="primary" variant="subtle" size="xs">
              {{ row.original.category }}
            </UBadge>
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
            Showing 1-10 of 128 articles
          </p>
          <UPagination v-model:page="currentPage" :total="128" :items-per-page="10" />
        </div>
      </UCard>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false
})

const currentPage = ref(1)

const statusOptions = [
  { label: 'All Status', value: 'all' },
  { label: 'Published', value: 'published' },
  { label: 'Draft', value: 'draft' }
]

const categoryOptions = [
  { label: 'All Categories', value: 'all' },
  { label: 'Technology', value: 'technology' },
  { label: 'Science', value: 'science' },
  { label: 'Business', value: 'business' },
  { label: 'Health', value: 'health' }
]

const columns = [
  { key: 'title', header: 'Article', size: 300 },
  { key: 'category', header: 'Category' },
  { key: 'difficulty', header: 'Difficulty' },
  { key: 'views', header: 'Views' },
  { key: 'status', header: 'Status' },
  { key: 'createdAt', header: 'Created' },
  { key: 'actions', header: '' }
]

const articles = [
  {
    id: 1,
    title: 'The Future of Artificial Intelligence in Healthcare',
    excerpt: 'Explore how AI is revolutionizing...',
    category: 'Technology',
    difficulty: 'Intermediate',
    views: '2.3k',
    status: 'published',
    createdAt: 'Mar 5, 2026',
    cover: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=100&h=100&fit=crop'
  },
  {
    id: 2,
    title: 'Climate Change: What Scientists Are Saying',
    excerpt: 'Understanding the latest research...',
    category: 'Science',
    difficulty: 'Advanced',
    views: '1.8k',
    status: 'published',
    createdAt: 'Mar 4, 2026',
    cover: 'https://images.unsplash.com/photo-1569163139599-0f4517e36f51?w=100&h=100&fit=crop'
  },
  {
    id: 3,
    title: 'Building a Successful Startup',
    excerpt: 'Key insights from entrepreneurs...',
    category: 'Business',
    difficulty: 'Beginner',
    views: '3.1k',
    status: 'draft',
    createdAt: 'Mar 3, 2026',
    cover: 'https://images.unsplash.com/photo-1559136555-9303baea8ebd?w=100&h=100&fit=crop'
  },
  {
    id: 4,
    title: 'The Science of Sleep: Why It Matters',
    excerpt: 'Discover how quality sleep affects...',
    category: 'Health',
    difficulty: 'Beginner',
    views: '4.2k',
    status: 'published',
    createdAt: 'Mar 2, 2026',
    cover: 'https://images.unsplash.com/photo-1541781774459-bb2af2f05b55?w=100&h=100&fit=crop'
  },
  {
    id: 5,
    title: 'Digital Transformation in Modern Business',
    excerpt: 'How companies are adapting...',
    category: 'Business',
    difficulty: 'Intermediate',
    views: '1.5k',
    status: 'draft',
    createdAt: 'Mar 1, 2026',
    cover: 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=100&h=100&fit=crop'
  }
]

const difficultyColor = (difficulty: string) => {
  switch (difficulty) {
    case 'Beginner': return 'success'
    case 'Intermediate': return 'warning'
    case 'Advanced': return 'error'
    default: return 'neutral'
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
    to: `/articles/${article.id}`
  }],
  [{
    label: 'Delete',
    icon: 'i-lucide-trash-2',
    color: 'error'
  }]
]
</script>