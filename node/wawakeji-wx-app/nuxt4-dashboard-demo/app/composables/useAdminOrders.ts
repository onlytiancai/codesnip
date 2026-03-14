import type { Ref } from 'vue'

interface User {
  id: number
  email: string
  name: string | null
  avatar: string | null
}

interface Order {
  id: number
  orderNo: string
  userId: number
  plan: string
  amount: number
  status: string
  paymentMethod: string
  transactionId: string | null
  paidAt: string | null
  createdAt: string
  expiredAt: string
  User: User
}

interface Pagination {
  page: number
  limit: number
  total: number
  totalPages: number
}

interface OrderStats {
  totalOrders: number
  pendingOrders: number
  paidOrders: number
  failedOrders: number
  refundedOrders: number
  totalRevenue: number
  todayRevenue: number
  monthRevenue: number
  recentOrders: Order[]
  planDistribution: { plan: string; _count: number; _sum: { amount: number | null } }[]
}

export const useAdminOrders = () => {
  const orders: Ref<Order[]> = ref([])
  const pagination: Ref<Pagination> = ref({
    page: 1,
    limit: 10,
    total: 0,
    totalPages: 0
  })
  const stats: Ref<OrderStats | null> = ref(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Fetch orders with pagination and filters
  const fetchOrders = async (filters?: {
    page?: number
    limit?: number
    status?: string
    plan?: string
    search?: string
  }) => {
    loading.value = true
    error.value = null

    try {
      const query = new URLSearchParams()
      if (filters?.page) query.set('page', filters.page.toString())
      if (filters?.limit) query.set('limit', filters.limit.toString())
      if (filters?.status) query.set('status', filters.status)
      if (filters?.plan) query.set('plan', filters.plan)
      if (filters?.search) query.set('search', filters.search)

      const response = await $fetch<{ orders: Order[]; pagination: Pagination }>(
        `/api/admin/orders?${query.toString()}`
      )
      orders.value = response.orders
      pagination.value = response.pagination
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch orders'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Fetch order statistics
  const fetchStats = async () => {
    try {
      const response = await $fetch<OrderStats>('/api/admin/orders/stats')
      stats.value = response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch order stats'
      throw e
    }
  }

  // Format amount from cents to yuan
  const formatAmount = (amount: number) => {
    return (amount / 100).toFixed(2)
  }

  return {
    orders,
    pagination,
    stats,
    loading,
    error,
    fetchOrders,
    fetchStats,
    formatAmount
  }
}