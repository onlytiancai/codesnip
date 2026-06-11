<template>
  <NuxtLayout name="default">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Header -->
      <div class="flex items-center justify-between mb-8">
        <div>
          <h1 class="text-2xl font-bold">Bookmarks</h1>
          <p class="text-gray-500 dark:text-gray-400">Articles you've saved for later</p>
        </div>
        <div class="flex items-center gap-2">
          <UInput
            v-model="searchQuery"
            placeholder="Search bookmarks..."
            icon="i-lucide-search"
            size="sm"
            class="w-48"
          />
        </div>
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="flex justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
      </div>

      <template v-else>
        <!-- Stats -->
        <div class="grid grid-cols-2 gap-4 mb-8">
          <UCard class="text-center">
            <p class="text-2xl font-bold text-primary">{{ pagination.total }}</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">Saved Articles</p>
          </UCard>
        </div>

        <!-- Bookmarks Grid -->
        <div v-if="filteredBookmarks.length > 0" class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div
            v-for="bookmark in filteredBookmarks"
            :key="bookmark.id"
            class="group"
          >
            <UCard class="hover:border-primary transition cursor-pointer h-full relative">
              <NuxtLink :to="`/articles/${bookmark.slug}`">
                <div class="flex gap-4">
                  <img
                    :src="bookmark.cover || '/placeholder.jpg'"
                    :alt="bookmark.title"
                    class="w-24 h-24 object-cover rounded-lg flex-shrink-0 bg-gray-100 dark:bg-gray-800"
                  />
                  <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 mb-1">
                      <UBadge v-if="bookmark.category" color="primary" variant="subtle" size="xs">
                        {{ bookmark.category.name }}
                      </UBadge>
                    </div>
                    <h4 class="font-medium line-clamp-2 group-hover:text-primary transition">
                      {{ bookmark.title }}
                    </h4>
                    <div class="flex items-center justify-between mt-2">
                      <span class="text-xs text-gray-500 dark:text-gray-400">
                        {{ bookmark.readTime }} min read
                      </span>
                    </div>
                  </div>
                </div>
              </NuxtLink>
              <UButton
                icon="i-lucide-bookmark"
                size="xs"
                variant="ghost"
                color="primary"
                class="absolute top-2 right-2"
                @click.prevent="handleRemove(bookmark.articleId)"
              />
            </UCard>
          </div>
        </div>

        <!-- Empty State -->
        <div v-else class="text-center py-12">
          <UIcon name="i-lucide-bookmark" class="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
          <h3 class="text-lg font-medium mb-2">No bookmarks yet</h3>
          <p class="text-gray-500 dark:text-gray-400 mb-4">
            Start saving articles to read them later
          </p>
          <UButton to="/articles">Browse Articles</UButton>
        </div>

        <!-- Pagination -->
        <div v-if="pagination.totalPages > 1" class="flex justify-center gap-2 mt-8">
          <UButton
            variant="outline"
            size="sm"
            :disabled="pagination.page === 1"
            @click="loadPage(pagination.page - 1)"
          >
            Previous
          </UButton>
          <span class="flex items-center px-4 text-sm text-gray-500">
            Page {{ pagination.page }} of {{ pagination.totalPages }}
          </span>
          <UButton
            variant="outline"
            size="sm"
            :disabled="pagination.page === pagination.totalPages"
            @click="loadPage(pagination.page + 1)"
          >
            Next
          </UButton>
        </div>
      </template>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  middleware: 'auth'
})

const { bookmarks, pagination, loading, fetchBookmarks, removeBookmark } = useBookmarks()
const toast = useToast()
const searchQuery = ref('')

onMounted(() => {
  fetchBookmarks()
})

const filteredBookmarks = computed(() => {
  if (!searchQuery.value) return bookmarks.value

  const query = searchQuery.value.toLowerCase()
  return bookmarks.value.filter(b =>
    b.title.toLowerCase().includes(query) ||
    b.category?.name.toLowerCase().includes(query)
  )
})

const loadPage = async (page: number) => {
  await fetchBookmarks({ page })
}

const handleRemove = async (articleId: number) => {
  try {
    await removeBookmark(articleId)
    toast.add({
      title: 'Bookmark removed',
      color: 'success'
    })
  } catch (error) {
    toast.add({
      title: 'Failed to remove bookmark',
      color: 'error'
    })
  }
}
</script>