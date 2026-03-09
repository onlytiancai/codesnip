import type { Ref } from 'vue'

interface Category {
  id: number
  name: string
  slug: string
  description: string | null
  icon: string | null
  color: string
  status: string
  sortOrder: number
  articleCount: number
  createdAt: string
  updatedAt: string
}

interface CategoryForm {
  name: string
  slug: string
  description?: string
  icon?: string
  color?: string
  status?: string
  sortOrder?: number
}

export const useAdminCategories = () => {
  const categories: Ref<Category[]> = ref([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Fetch all categories
  const fetchCategories = async (filters?: { status?: string }) => {
    loading.value = true
    error.value = null

    try {
      const query = new URLSearchParams()
      if (filters?.status) {
        query.set('status', filters.status)
      }

      const response = await $fetch<Category[]>(`/api/admin/categories?${query.toString()}`)
      categories.value = response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch categories'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Create category
  const createCategory = async (data: CategoryForm) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<Category>('/api/admin/categories', {
        method: 'POST',
        body: data
      })
      categories.value.push(response)
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to create category'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Update category
  const updateCategory = async (id: number, data: Partial<CategoryForm>) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<Category>(`/api/admin/categories/${id}`, {
        method: 'PUT',
        body: data
      })
      const index = categories.value.findIndex(c => c.id === id)
      if (index !== -1) {
        categories.value[index] = { ...categories.value[index], ...response }
      }
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to update category'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Delete category
  const deleteCategory = async (id: number) => {
    loading.value = true
    error.value = null

    try {
      await $fetch(`/api/admin/categories/${id}`, {
        method: 'DELETE'
      })
      categories.value = categories.value.filter(c => c.id !== id)
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to delete category'
      throw e
    } finally {
      loading.value = false
    }
  }

  return {
    categories,
    loading,
    error,
    fetchCategories,
    createCategory,
    updateCategory,
    deleteCategory
  }
}