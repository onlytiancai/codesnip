<template>
  <NuxtLayout name="admin">
    <div>
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-2xl font-bold">Orders</h2>
      </div>

      <!-- Stats -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
        <UCard class="text-center">
          <p class="text-2xl font-bold">{{ stats?.totalOrders || 0 }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Total Orders</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-yellow-500">{{ stats?.pendingOrders || 0 }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Pending</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-green-500">{{ stats?.paidOrders || 0 }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Paid</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-blue-500">¥{{ formatAmount(stats?.monthRevenue || 0) }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">This Month</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-purple-500">¥{{ formatAmount(stats?.totalRevenue || 0) }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Total Revenue</p>
        </UCard>
      </div>

      <!-- Filters -->
      <div class="flex flex-wrap items-center gap-3 mb-6">
        <UInput
          v-model="searchQuery"
          placeholder="Search by order no, user..."
          icon="i-lucide-search"
          class="w-64"
          @input="debouncedSearch"
        />
        <USelect
          v-model="statusFilter"
          :items="statusOptions"
          placeholder="Status"
          class="w-32"
          @change="applyFilters"
        />
        <USelect
          v-model="planFilter"
          :items="planOptions"
          placeholder="Plan"
          class="w-32"
          @change="applyFilters"
        />
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="flex items-center justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
      </div>

      <!-- Orders Table -->
      <UCard v-else>
        <UTable :data="orders" :columns="columns">
          <template #orderNo-cell="{ row }">
            <div>
              <p class="font-mono text-sm">{{ row.original.orderNo }}</p>
              <p class="text-xs text-gray-500">{{ formatDate(row.original.createdAt) }}</p>
            </div>
          </template>
          <template #user-cell="{ row }">
            <div class="flex items-center gap-3">
              <UAvatar :src="row.original.User?.avatar || undefined" :alt="row.original.User?.name || row.original.User?.email" size="sm" />
              <div>
                <p class="font-medium">{{ row.original.User?.name || 'No name' }}</p>
                <p class="text-xs text-gray-500 dark:text-gray-400">{{ row.original.User?.email }}</p>
              </div>
            </div>
          </template>
          <template #plan-cell="{ row }">
            <UBadge
              :color="row.original.plan === 'annual' ? 'success' : 'primary'"
              variant="subtle"
              size="xs"
            >
              {{ row.original.plan === 'annual' ? 'Annual' : 'Pro' }}
            </UBadge>
          </template>
          <template #amount-cell="{ row }">
            <span class="font-medium">¥{{ formatAmount(row.original.amount) }}</span>
          </template>
          <template #status-cell="{ row }">
            <UBadge
              :color="getStatusColor(row.original.status)"
              variant="subtle"
              size="xs"
            >
              {{ row.original.status }}
            </UBadge>
          </template>
          <template #paidAt-cell="{ row }">
            <span v-if="row.original.paidAt" class="text-sm">
              {{ formatDate(row.original.paidAt) }}
            </span>
            <span v-else class="text-gray-400">-</span>
          </template>
        </UTable>

        <!-- Pagination -->
        <div class="flex items-center justify-between mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <p class="text-sm text-gray-500 dark:text-gray-400">
            Showing {{ (pagination.page - 1) * pagination.limit + 1 }}-{{ Math.min(pagination.page * pagination.limit, pagination.total) }} of {{ pagination.total }} orders
          </p>
          <UPagination
            v-model:page="currentPage"
            :total="pagination.total"
            :items-per-page="pagination.limit"
            @update:page="handlePageChange"
          />
        </div>
      </UCard>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false,
  middleware: 'admin'
})

const {
  orders,
  pagination,
  stats,
  loading,
  fetchOrders,
  fetchStats,
  formatAmount
} = useAdminOrders()

const currentPage = ref(1)
const searchQuery = ref('')
const statusFilter = ref('all')
const planFilter = ref('all')

const statusOptions = [
  { label: 'All Status', value: 'all' },
  { label: 'Pending', value: 'pending' },
  { label: 'Paid', value: 'paid' },
  { label: 'Failed', value: 'failed' },
  { label: 'Refunded', value: 'refunded' }
]

const planOptions = [
  { label: 'All Plans', value: 'all' },
  { label: 'Pro', value: 'pro' },
  { label: 'Annual', value: 'annual' }
]

const columns = [
  { id: 'orderNo', header: 'Order No' },
  { id: 'user', header: 'User' },
  { id: 'plan', header: 'Plan' },
  { id: 'amount', header: 'Amount' },
  { id: 'status', header: 'Status' },
  { id: 'paidAt', header: 'Paid At' }
]

const getStatusColor = (status: string) => {
  const colors: Record<string, string> = {
    pending: 'warning',
    paid: 'success',
    failed: 'error',
    refunded: 'neutral'
  }
  return colors[status] || 'neutral'
}

const formatDate = (dateStr: string) => {
  const date = new Date(dateStr)
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

const applyFilters = () => {
  currentPage.value = 1
  fetchData()
}

const handlePageChange = (page: number) => {
  currentPage.value = page
  fetchData()
}

const fetchData = () => {
  fetchOrders({
    page: currentPage.value,
    status: statusFilter.value !== 'all' ? statusFilter.value : undefined,
    plan: planFilter.value !== 'all' ? planFilter.value : undefined,
    search: searchQuery.value || undefined
  })
}

// Debounced search
let searchTimeout: NodeJS.Timeout
const debouncedSearch = () => {
  clearTimeout(searchTimeout)
  searchTimeout = setTimeout(() => {
    currentPage.value = 1
    fetchData()
  }, 300)
}

onMounted(() => {
  fetchData()
  fetchStats()
})
</script>