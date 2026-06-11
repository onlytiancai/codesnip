import type { Ref } from 'vue'

interface VocabularyItem {
  id: number
  word: string
  phonetic: string | null
  definition: string
  example: string | null
  progress: number
  article: { id: number; title: string; slug: string } | null
  createdAt: string
  lastReviewAt: string | null
}

interface VocabularyStats {
  totalWords: number
  mastered: number
  learning: number
}

interface Pagination {
  page: number
  limit: number
  total: number
  totalPages: number
}

export const useVocabulary = () => {
  const vocabulary: Ref<VocabularyItem[]> = ref([])
  const stats: Ref<VocabularyStats | null> = ref(null)
  const pagination: Ref<Pagination> = ref({
    page: 1,
    limit: 20,
    total: 0,
    totalPages: 0
  })
  const loading = ref(false)
  const error = ref<string | null>(null)

  const fetchVocabulary = async (options?: {
    page?: number
    limit?: number
    filter?: string
    sort?: string
  }) => {
    loading.value = true
    error.value = null

    try {
      const query = new URLSearchParams()
      if (options?.page) query.set('page', options.page.toString())
      if (options?.limit) query.set('limit', options.limit.toString())
      if (options?.filter) query.set('filter', options.filter)
      if (options?.sort) query.set('sort', options.sort)

      const response = await $fetch<{
        vocabulary: VocabularyItem[]
        pagination: Pagination
        stats: VocabularyStats
      }>(`/api/user/vocabulary?${query.toString()}`)

      vocabulary.value = response.vocabulary
      pagination.value = response.pagination
      stats.value = response.stats
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch vocabulary'
      throw e
    } finally {
      loading.value = false
    }
  }

  const addWord = async (data: {
    word: string
    phonetic?: string
    definition: string
    example?: string
    articleId?: number
  }) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<{ vocabulary: VocabularyItem }>('/api/user/vocabulary', {
        method: 'POST',
        body: data
      })
      vocabulary.value.unshift(response.vocabulary)
      if (stats.value) {
        stats.value.totalWords++
        stats.value.learning++
      }
      return response.vocabulary
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to add word'
      throw e
    } finally {
      loading.value = false
    }
  }

  const updateWord = async (id: number, data: {
    progress?: number
    phonetic?: string
    definition?: string
    example?: string
  }) => {
    try {
      const response = await $fetch<{ vocabulary: VocabularyItem }>(`/api/user/vocabulary/${id}`, {
        method: 'PUT',
        body: data
      })

      const index = vocabulary.value.findIndex(v => v.id === id)
      if (index !== -1) {
        vocabulary.value[index] = response.vocabulary
      }
      return response.vocabulary
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to update word'
      throw e
    }
  }

  const deleteWord = async (id: number) => {
    try {
      await $fetch(`/api/user/vocabulary/${id}`, {
        method: 'DELETE'
      })
      vocabulary.value = vocabulary.value.filter(v => v.id !== id)
      if (stats.value) {
        stats.value.totalWords--
      }
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to delete word'
      throw e
    }
  }

  return {
    vocabulary,
    stats,
    pagination,
    loading,
    error,
    fetchVocabulary,
    addWord,
    updateWord,
    deleteWord
  }
}