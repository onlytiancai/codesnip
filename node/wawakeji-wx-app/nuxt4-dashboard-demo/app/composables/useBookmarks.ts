import type { Ref } from 'vue'

interface BookmarkItem {
  id: number
  articleId: number
  title: string
  slug: string
  cover: string | null
  excerpt: string | null
  difficulty: string
  category: { id: number; name: string; slug: string } | null
  createdAt: string
  readTime: number
}

interface Pagination {
  page: number
  limit: number
  total: number
  totalPages: number
}

export const useBookmarks = () => {
  const bookmarks: Ref<BookmarkItem[]> = ref([])
  const pagination: Ref<Pagination> = ref({
    page: 1,
    limit: 10,
    total: 0,
    totalPages: 0
  })
  const loading = ref(false)
  const error = ref<string | null>(null)
  const bookmarkedIds: Ref<Set<number>> = ref(new Set())

  const fetchBookmarks = async (options?: { page?: number; limit?: number }) => {
    loading.value = true
    error.value = null

    try {
      const query = new URLSearchParams()
      if (options?.page) query.set('page', options.page.toString())
      if (options?.limit) query.set('limit', options.limit.toString())

      const response = await $fetch<{
        bookmarks: BookmarkItem[]
        pagination: Pagination
      }>(`/api/user/bookmarks?${query.toString()}`)

      bookmarks.value = response.bookmarks
      pagination.value = response.pagination

      // Update bookmarked ids set
      response.bookmarks.forEach(b => bookmarkedIds.value.add(b.articleId))
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch bookmarks'
      throw e
    } finally {
      loading.value = false
    }
  }

  const addBookmark = async (articleId: number) => {
    try {
      await $fetch(`/api/user/bookmarks/${articleId}`, {
        method: 'POST'
      })
      bookmarkedIds.value.add(articleId)
      return true
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to add bookmark'
      throw e
    }
  }

  const removeBookmark = async (articleId: number) => {
    try {
      await $fetch(`/api/user/bookmarks/${articleId}`, {
        method: 'DELETE'
      })
      bookmarkedIds.value.delete(articleId)
      bookmarks.value = bookmarks.value.filter(b => b.articleId !== articleId)
      pagination.value.total = Math.max(0, pagination.value.total - 1)
      return true
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to remove bookmark'
      throw e
    }
  }

  const isBookmarked = (articleId: number) => {
    return bookmarkedIds.value.has(articleId)
  }

  const toggleBookmark = async (articleId: number) => {
    if (isBookmarked(articleId)) {
      return removeBookmark(articleId)
    } else {
      return addBookmark(articleId)
    }
  }

  return {
    bookmarks,
    pagination,
    loading,
    error,
    bookmarkedIds,
    fetchBookmarks,
    addBookmark,
    removeBookmark,
    isBookmarked,
    toggleBookmark
  }
}