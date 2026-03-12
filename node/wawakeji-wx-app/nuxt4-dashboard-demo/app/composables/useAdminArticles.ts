import type { Ref } from 'vue'

interface Article {
  id: number
  title: string
  slug: string
  excerpt: string | null
  cover: string | null
  content: string | null
  status: string
  difficulty: string
  views: number
  bookmarks: number
  publishAt: string | null
  metaTitle: string | null
  metaDesc: string | null
  categoryId: number | null
  category: { id: number; name: string; slug: string } | null
  tags: { id: number; name: string; slug: string; color: string }[]
  sentences: { id: number; order: number; en: string; cn: string | null; audio: string | null }[]
  author: { id: number; name: string | null; email: string }
  sentenceCount: number
  createdAt: string
  updatedAt: string
}

interface ArticleForm {
  title: string
  slug: string
  excerpt?: string
  cover?: string
  content?: string
  status?: string
  difficulty?: string
  publishAt?: string
  metaTitle?: string
  metaDesc?: string
  categoryId?: number | null
  tagIds?: number[]
  sentences?: { order: number; en: string; cn?: string; audio?: string }[]
}

interface Pagination {
  page: number
  limit: number
  total: number
  totalPages: number
}

export const useAdminArticles = () => {
  const articles: Ref<Article[]> = ref([])
  const article: Ref<Article | null> = ref(null)
  const pagination: Ref<Pagination> = ref({
    page: 1,
    limit: 10,
    total: 0,
    totalPages: 0
  })
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Fetch articles with pagination and filters
  const fetchArticles = async (filters?: {
    page?: number
    limit?: number
    status?: string
    categoryId?: string
    difficulty?: string
    search?: string
  }) => {
    loading.value = true
    error.value = null

    try {
      const query = new URLSearchParams()
      if (filters?.page) query.set('page', filters.page.toString())
      if (filters?.limit) query.set('limit', filters.limit.toString())
      if (filters?.status) query.set('status', filters.status)
      if (filters?.categoryId) query.set('categoryId', filters.categoryId)
      if (filters?.difficulty) query.set('difficulty', filters.difficulty)
      if (filters?.search) query.set('search', filters.search)

      const response = await $fetch<{ articles: Article[]; pagination: Pagination }>(
        `/api/admin/articles?${query.toString()}`
      )
      articles.value = response.articles
      pagination.value = response.pagination
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch articles'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Fetch single article
  const fetchArticle = async (id: number) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<Article>(`/api/admin/articles/${id}`)
      article.value = response
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch article'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Create article
  const createArticle = async (data: ArticleForm) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<Article>('/api/admin/articles', {
        method: 'POST',
        body: data
      })
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to create article'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Update article
  const updateArticle = async (id: number, data: Partial<ArticleForm>) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<Article>(`/api/admin/articles/${id}`, {
        method: 'PUT',
        body: data
      })
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to update article'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Delete article
  const deleteArticle = async (id: number) => {
    loading.value = true
    error.value = null

    try {
      await $fetch(`/api/admin/articles/${id}`, {
        method: 'DELETE'
      })
      articles.value = articles.value.filter(a => a.id !== id)
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to delete article'
      throw e
    } finally {
      loading.value = false
    }
  }

  return {
    articles,
    article,
    pagination,
    loading,
    error,
    fetchArticles,
    fetchArticle,
    createArticle,
    updateArticle,
    deleteArticle
  }
}