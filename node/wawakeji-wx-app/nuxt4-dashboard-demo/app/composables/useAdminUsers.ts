import type { Ref } from 'vue'

interface User {
  id: number
  email: string
  name: string | null
  avatar: string | null
  role: string
  articleCount: number
  createdAt: string
  updatedAt: string
}

interface Pagination {
  page: number
  limit: number
  total: number
  totalPages: number
}

export const useAdminUsers = () => {
  const users: Ref<User[]> = ref([])
  const pagination: Ref<Pagination> = ref({
    page: 1,
    limit: 10,
    total: 0,
    totalPages: 0
  })
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Fetch users with pagination and filters
  const fetchUsers = async (filters?: {
    page?: number
    limit?: number
    role?: string
    search?: string
  }) => {
    loading.value = true
    error.value = null

    try {
      const query = new URLSearchParams()
      if (filters?.page) query.set('page', filters.page.toString())
      if (filters?.limit) query.set('limit', filters.limit.toString())
      if (filters?.role) query.set('role', filters.role)
      if (filters?.search) query.set('search', filters.search)

      const response = await $fetch<{ users: User[]; pagination: Pagination }>(
        `/api/admin/users?${query.toString()}`
      )
      users.value = response.users
      pagination.value = response.pagination
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch users'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Fetch single user
  const fetchUser = async (id: number) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<User>(`/api/admin/users/${id}`)
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch user'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Update user
  const updateUser = async (id: number, data: { name?: string; avatar?: string; role?: string }) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<User>(`/api/admin/users/${id}`, {
        method: 'PUT',
        body: data
      })
      const index = users.value.findIndex(u => u.id === id)
      if (index !== -1) {
        users.value[index] = { ...users.value[index], ...response }
      }
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to update user'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Delete user
  const deleteUser = async (id: number) => {
    loading.value = true
    error.value = null

    try {
      await $fetch(`/api/admin/users/${id}`, {
        method: 'DELETE'
      })
      users.value = users.value.filter(u => u.id !== id)
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to delete user'
      throw e
    } finally {
      loading.value = false
    }
  }

  return {
    users,
    pagination,
    loading,
    error,
    fetchUsers,
    fetchUser,
    updateUser,
    deleteUser
  }
}