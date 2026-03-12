// Admin Dictionary Management Composable
import type { Ref } from 'vue'

interface DictionaryWord {
  id: number
  word: string
  phonetic: string | null
  definition: string | null
  translation: string | null
  pos: string | null
  collins: number | null
  oxford: number | null
  tag: string | null
  bnc: number | null
  frq: number | null
  exchange: string | null
}

interface DictionaryStats {
  total: number
  withPhonetic: number
  withTranslation: number
  withDefinition: number
  collins5Star: number
  oxford3000: number
  tagStats: Array<{ tag: string; label: string; count: number }>
}

interface DictionaryFilters {
  search?: string
  tag?: string
  hasPhonetic?: 'true' | 'false' | 'all'
  hasTranslation?: 'true' | 'false' | 'all'
  sortBy?: 'word' | 'collins' | 'bnc' | 'frq'
  sortOrder?: 'asc' | 'desc'
}

export const useAdminDictionary = () => {
  const toast = useToast()
  const words: Ref<DictionaryWord[]> = ref([])
  const stats: Ref<DictionaryStats | null> = ref(null)
  const loading = ref(false)
  const statsLoading = ref(false)
  const pagination = ref({
    page: 1,
    limit: 20,
    total: 0,
    totalPages: 0
  })
  const filters = ref<DictionaryFilters>({
    search: '',
    tag: '',
    hasPhonetic: 'all',
    hasTranslation: 'all',
    sortBy: 'word',
    sortOrder: 'asc'
  })

  const fetchWords = async (page = 1) => {
    loading.value = true
    try {
      const query: Record<string, string | number> = {
        page,
        limit: pagination.value.limit,
        sortBy: filters.value.sortBy || 'word',
        sortOrder: filters.value.sortOrder || 'asc'
      }
      if (filters.value.search) query.search = filters.value.search
      if (filters.value.tag) query.tag = filters.value.tag
      if (filters.value.hasPhonetic && filters.value.hasPhonetic !== 'all') {
        query.hasPhonetic = filters.value.hasPhonetic
      }
      if (filters.value.hasTranslation && filters.value.hasTranslation !== 'all') {
        query.hasTranslation = filters.value.hasTranslation
      }

      const response = await $fetch('/api/admin/dictionary', { query })
      words.value = response.words
      pagination.value = { ...pagination.value, ...response.pagination }
    } catch (error: any) {
      toast.add({
        title: 'Failed to fetch words',
        description: error.data?.message || 'Please try again',
        color: 'error'
      })
    } finally {
      loading.value = false
    }
  }

  const fetchStats = async () => {
    statsLoading.value = true
    try {
      stats.value = await $fetch('/api/admin/dictionary/stats')
    } catch (error: any) {
      toast.add({
        title: 'Failed to fetch statistics',
        description: error.data?.message || 'Please try again',
        color: 'error'
      })
    } finally {
      statsLoading.value = false
    }
  }

  const createWord = async (data: Partial<DictionaryWord>) => {
    try {
      await $fetch('/api/admin/dictionary', {
        method: 'POST',
        body: data
      })
      toast.add({
        title: 'Word created',
        description: `"${data.word}" has been added to the dictionary`,
        color: 'success'
      })
      await fetchWords(pagination.value.page)
      await fetchStats()
      return true
    } catch (error: any) {
      toast.add({
        title: 'Failed to create word',
        description: error.data?.message || 'Please try again',
        color: 'error'
      })
      return false
    }
  }

  const updateWord = async (word: string, data: Partial<DictionaryWord>) => {
    try {
      await $fetch(`/api/admin/dictionary/${encodeURIComponent(word)}`, {
        method: 'PUT',
        body: data
      })
      toast.add({
        title: 'Word updated',
        description: `"${word}" has been updated`,
        color: 'success'
      })
      await fetchWords(pagination.value.page)
      return true
    } catch (error: any) {
      toast.add({
        title: 'Failed to update word',
        description: error.data?.message || 'Please try again',
        color: 'error'
      })
      return false
    }
  }

  const deleteWord = async (word: string) => {
    try {
      await $fetch(`/api/admin/dictionary/${encodeURIComponent(word)}`, {
        method: 'DELETE'
      })
      toast.add({
        title: 'Word deleted',
        description: `"${word}" has been removed from the dictionary`,
        color: 'success'
      })
      await fetchWords(pagination.value.page)
      await fetchStats()
      return true
    } catch (error: any) {
      toast.add({
        title: 'Failed to delete word',
        description: error.data?.message || 'Please try again',
        color: 'error'
      })
      return false
    }
  }

  const searchWords = async (query: string) => {
    filters.value.search = query
    pagination.value.page = 1
    await fetchWords(1)
  }

  const applyFilters = async () => {
    pagination.value.page = 1
    await fetchWords(1)
  }

  return {
    words,
    stats,
    loading,
    statsLoading,
    pagination,
    filters,
    fetchWords,
    fetchStats,
    createWord,
    updateWord,
    deleteWord,
    searchWords,
    applyFilters
  }
}