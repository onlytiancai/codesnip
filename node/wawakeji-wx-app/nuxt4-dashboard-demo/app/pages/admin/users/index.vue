<template>
  <NuxtLayout name="admin">
    <div>
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-2xl font-bold">Users</h2>
        <UButton icon="i-lucide-download" variant="outline">
          Export Users
        </UButton>
      </div>

      <!-- Stats -->
      <div class="grid grid-cols-1 sm:grid-cols-4 gap-4 mb-6">
        <UCard class="text-center">
          <p class="text-2xl font-bold">{{ pagination.total }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Total Users</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-purple-500">{{ adminCount }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Admins</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-blue-500">{{ userCount }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Regular Users</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-green-500">{{ totalArticles }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Total Articles</p>
        </UCard>
      </div>

      <!-- Filters -->
      <div class="flex flex-wrap items-center gap-3 mb-6">
        <UInput
          v-model="searchQuery"
          placeholder="Search users..."
          icon="i-lucide-search"
          class="w-64"
          @input="debouncedSearch"
        />
        <USelect
          v-model="roleFilter"
          :items="roleOptions"
          placeholder="Role"
          class="w-32"
          @change="applyFilters"
        />
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="flex items-center justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
      </div>

      <!-- Users Table -->
      <UCard v-else>
        <UTable :data="users" :columns="columns">
          <template #user-cell="{ row }">
            <div class="flex items-center gap-3">
              <UAvatar :src="row.original.avatar || undefined" :alt="row.original.name || row.original.email" size="sm" />
              <div>
                <p class="font-medium">{{ row.original.name || 'No name' }}</p>
                <p class="text-xs text-gray-500 dark:text-gray-400">{{ row.original.email }}</p>
              </div>
            </div>
          </template>
          <template #role-cell="{ row }">
            <UBadge
              :color="row.original.role === 'ADMIN' ? 'warning' : 'neutral'"
              variant="subtle"
              size="xs"
            >
              {{ row.original.role }}
            </UBadge>
          </template>
          <template #actions-cell="{ row }">
            <UDropdownMenu :items="getActionItems(row.original)">
              <UButton icon="i-lucide-more-horizontal" color="neutral" variant="ghost" size="xs" />
            </UDropdownMenu>
          </template>
        </UTable>

        <!-- Pagination -->
        <div class="flex items-center justify-between mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <p class="text-sm text-gray-500 dark:text-gray-400">
            Showing {{ (pagination.page - 1) * pagination.limit + 1 }}-{{ Math.min(pagination.page * pagination.limit, pagination.total) }} of {{ pagination.total }} users
          </p>
          <UPagination
            v-model:page="currentPage"
            :total="pagination.total"
            :items-per-page="pagination.limit"
            @update:page="handlePageChange"
          />
        </div>
      </UCard>

      <!-- Edit User Modal -->
      <UModal
        v-model:open="showEditModal"
        title="Edit User"
        description="Update user information"
      >
        <template #body>
          <div class="space-y-4">
            <UFormField label="Name" name="name">
              <UInput v-model="editForm.name" placeholder="User name" />
            </UFormField>
            <UFormField label="Role" name="role">
              <USelect
                v-model="editForm.role"
                :items="[
                  { label: 'User', value: 'USER' },
                  { label: 'Admin', value: 'ADMIN' }
                ]"
              />
            </UFormField>
          </div>
        </template>
        <template #footer>
          <div class="flex justify-end gap-3">
            <UButton variant="outline" @click="showEditModal = false">Cancel</UButton>
            <UButton color="primary" :loading="saving" @click="handleUpdateUser">Update</UButton>
          </div>
        </template>
      </UModal>

      <!-- Delete Confirmation -->
      <UModal v-model:open="showDeleteModal" title="Delete User" description="Are you sure you want to delete this user?">
        <template #body>
          <p class="text-gray-500">
            This action cannot be undone. The user "{{ userToDelete?.name || userToDelete?.email }}" will be permanently deleted.
            <span v-if="userToDelete?.articleCount" class="text-red-500 block mt-2">
              Warning: This user has {{ userToDelete.articleCount }} articles. They will also be deleted.
            </span>
          </p>
        </template>
        <template #footer>
          <div class="flex justify-end gap-3">
            <UButton variant="outline" @click="showDeleteModal = false">Cancel</UButton>
            <UButton color="error" :loading="deleting" @click="handleDeleteUser">Delete</UButton>
          </div>
        </template>
      </UModal>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false,
  middleware: 'admin'
})

const {
  users,
  pagination,
  loading,
  fetchUsers,
  updateUser,
  deleteUser
} = useAdminUsers()

const currentPage = ref(1)
const searchQuery = ref('')
const roleFilter = ref('all')

const showEditModal = ref(false)
const showDeleteModal = ref(false)
const userToEdit = ref<any>(null)
const userToDelete = ref<any>(null)
const saving = ref(false)
const deleting = ref(false)

const editForm = ref({
  name: '',
  role: 'USER'
})

const roleOptions = [
  { label: 'All Roles', value: 'all' },
  { label: 'Admin', value: 'ADMIN' },
  { label: 'User', value: 'USER' }
]

const columns = [
  { id: 'user', header: 'User' },
  { id: 'role', header: 'Role' },
  { id: 'articleCount', header: 'Articles', accessorKey: 'articleCount' },
  { id: 'createdAt', header: 'Joined', accessorKey: 'createdAt' },
  { id: 'actions', header: '' }
]

const adminCount = computed(() => users.value.filter(u => u.role === 'ADMIN').length)
const userCount = computed(() => users.value.filter(u => u.role === 'USER').length)
const totalArticles = computed(() => users.value.reduce((sum, u) => sum + u.articleCount, 0))

const applyFilters = () => {
  currentPage.value = 1
  fetchData()
}

const handlePageChange = (page: number) => {
  currentPage.value = page
  fetchData()
}

const fetchData = () => {
  fetchUsers({
    page: currentPage.value,
    role: roleFilter.value !== 'all' ? roleFilter.value : undefined,
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

const openEditModal = (user: any) => {
  userToEdit.value = user
  editForm.value = {
    name: user.name || '',
    role: user.role
  }
  showEditModal.value = true
}

const handleUpdateUser = async () => {
  if (!userToEdit.value) return

  saving.value = true
  try {
    await updateUser(userToEdit.value.id, editForm.value)
    showEditModal.value = false
  } catch (e) {
    // Error is handled in the composable
  } finally {
    saving.value = false
  }
}

const confirmDelete = (user: any) => {
  userToDelete.value = user
  showDeleteModal.value = true
}

const handleDeleteUser = async () => {
  if (!userToDelete.value) return

  deleting.value = true
  try {
    await deleteUser(userToDelete.value.id)
    showDeleteModal.value = false
    userToDelete.value = null
  } catch (e) {
    // Error is handled in the composable
  } finally {
    deleting.value = false
  }
}

const getActionItems = (user: any) => [
  [{
    label: 'Edit',
    icon: 'i-lucide-edit',
    click: () => openEditModal(user)
  }],
  [{
    label: 'Delete',
    icon: 'i-lucide-trash-2',
    color: 'error' as const,
    click: () => confirmDelete(user)
  }]
]

onMounted(() => {
  fetchData()
})
</script>