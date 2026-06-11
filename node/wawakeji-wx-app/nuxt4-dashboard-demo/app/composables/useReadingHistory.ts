import type { Ref } from 'vue'

interface HistoryItem {
  id: number
  articleId: number
  title: string
  slug: string
  cover: string | null
  excerpt: string | null
  difficulty: string
  category: { id: number; name: string; slug: string } | null
  progress: number
  lastReadAt: string
  completedAt: string | null
  readTime: number
}

interface HistoryStats {
  articlesRead: number
  totalMinutes: number
  streak: number
  inProgress: number
  completedThisWeek: number
}

interface Pagination {
  page: number
  limit: number
  total: number
  totalPages: number
}

export const useReadingHistory = () => {
  const history: Ref<HistoryItem[]> = ref([])
  const stats: Ref<HistoryStats | null> = ref(null)
  const pagination: Ref<Pagination> = ref({
    page: 1,
    limit: 10,
    total: 0,
    totalPages: 0
  })
  const loading = ref(false)
  const error = ref<string | null>(null)

  const fetchHistory = async (options?: { page?: number; limit?: number }) => {
    loading.value = true
    error.value = null

    try {
      const query = new URLSearchParams()
      if (options?.page) query.set('page', options.page.toString())
      if (options?.limit) query.set('limit', options.limit.toString())

      const response = await $fetch<{
        history: HistoryItem[]
        pagination: Pagination
      }>(`/api/user/history?${query.toString()}`)

      history.value = response.history
      pagination.value = response.pagination
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch history'
      throw e
    } finally {
      loading.value = false
    }
  }

  const fetchStats = async () => {
    try {
      const response = await $fetch<HistoryStats>('/api/user/history/stats')
      stats.value = response
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch stats'
      throw e
    }
  }

  const updateProgress = async (articleId: number, progress: number) => {
    try {
      const response = await $fetch(`/api/user/history/${articleId}`, {
        method: 'POST',
        body: { progress }
      })
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to update progress'
      throw e
    }
  }

  return {
    history,
    stats,
    pagination,
    loading,
    error,
    fetchHistory,
    fetchStats,
    updateProgress
  }
}