import type { Ref } from 'vue'

interface Tag {
  id: number
  name: string
  slug: string
  description: string | null
  color: string
  articleCount: number
  createdAt: string
  updatedAt: string
}

interface TagForm {
  name: string
  slug: string
  description?: string
  color?: string
}

export const useAdminTags = () => {
  const tags: Ref<Tag[]> = ref([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Fetch all tags
  const fetchTags = async (filters?: { search?: string }) => {
    loading.value = true
    error.value = null

    try {
      const query = new URLSearchParams()
      if (filters?.search) {
        query.set('search', filters.search)
      }

      const response = await $fetch<Tag[]>(`/api/admin/tags?${query.toString()}`)
      tags.value = response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch tags'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Create tag
  const createTag = async (data: TagForm) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<Tag>('/api/admin/tags', {
        method: 'POST',
        body: data
      })
      tags.value.push(response)
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to create tag'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Update tag
  const updateTag = async (id: number, data: Partial<TagForm>) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<Tag>(`/api/admin/tags/${id}`, {
        method: 'PUT',
        body: data
      })
      const index = tags.value.findIndex(t => t.id === id)
      if (index !== -1) {
        tags.value[index] = { ...tags.value[index], ...response }
      }
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to update tag'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Delete tag
  const deleteTag = async (id: number) => {
    loading.value = true
    error.value = null

    try {
      await $fetch(`/api/admin/tags/${id}`, {
        method: 'DELETE'
      })
      tags.value = tags.value.filter(t => t.id !== id)
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to delete tag'
      throw e
    } finally {
      loading.value = false
    }
  }

  return {
    tags,
    loading,
    error,
    fetchTags,
    createTag,
    updateTag,
    deleteTag
  }
}