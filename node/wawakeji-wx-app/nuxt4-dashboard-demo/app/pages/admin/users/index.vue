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
          <p class="text-2xl font-bold">12,543</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Total Users</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-green-500">892</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Premium</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-blue-500">11,651</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Free Users</p>
        </UCard>
        <UCard class="text-center">
          <p class="text-2xl font-bold text-orange-500">156</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">New This Month</p>
        </UCard>
      </div>

      <!-- Filters -->
      <div class="flex flex-wrap items-center gap-3 mb-6">
        <UInput
          placeholder="Search users..."
          icon="i-lucide-search"
          class="w-64"
        />
        <USelect
          :items="roleOptions"
          placeholder="Role"
          class="w-32"
        />
        <USelect
          :items="statusOptions"
          placeholder="Status"
          class="w-32"
        />
        <USelect
          :items="membershipOptions"
          placeholder="Membership"
          class="w-36"
        />
      </div>

      <!-- Users Table -->
      <UCard>
        <UTable :data="users" :columns="columns">
          <template #user-cell="{ row }">
            <div class="flex items-center gap-3">
              <UAvatar :src="row.original.avatar" :alt="row.original.name" size="sm" />
              <div>
                <p class="font-medium">{{ row.original.name }}</p>
                <p class="text-xs text-gray-500 dark:text-gray-400">{{ row.original.email }}</p>
              </div>
            </div>
          </template>
          <template #membership-cell="{ row }">
            <UBadge
              :color="row.original.membership === 'premium' ? 'warning' : 'neutral'"
              variant="subtle"
              size="xs"
            >
              <UIcon v-if="row.original.membership === 'premium'" name="i-lucide-crown" class="w-3 h-3 mr-1" />
              {{ row.original.membership }}
            </UBadge>
          </template>
          <template #status-cell="{ row }">
            <UBadge
              :color="row.original.status === 'active' ? 'success' : 'error'"
              variant="subtle"
              size="xs"
            >
              {{ row.original.status }}
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
            Showing 1-10 of 12,543 users
          </p>
          <UPagination v-model:page="currentPage" :total="12543" :items-per-page="10" />
        </div>
      </UCard>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false
})

const currentPage = ref(1)

const roleOptions = [
  { label: 'All Roles', value: 'all' },
  { label: 'User', value: 'user' },
  { label: 'Admin', value: 'admin' }
]

const statusOptions = [
  { label: 'All Status', value: 'all' },
  { label: 'Active', value: 'active' },
  { label: 'Inactive', value: 'inactive' },
  { label: 'Banned', value: 'banned' }
]

const membershipOptions = [
  { label: 'All', value: 'all' },
  { label: 'Premium', value: 'premium' },
  { label: 'Free', value: 'free' }
]

const columns = [
  { id: 'user', header: 'User' },
  { id: 'membership', header: 'Membership' },
  { id: 'articlesRead', header: 'Articles Read' },
  { id: 'joinDate', header: 'Joined' },
  { id: 'lastActive', header: 'Last Active' },
  { id: 'status', header: 'Status' },
  { id: 'actions', header: '' }
]

const users = [
  {
    id: 1,
    name: 'John Doe',
    email: 'john.doe@example.com',
    avatar: 'https://avatars.githubusercontent.com/u/1?v=4',
    membership: 'premium',
    articlesRead: 156,
    joinDate: 'Jan 15, 2026',
    lastActive: '2h ago',
    status: 'active'
  },
  {
    id: 2,
    name: 'Jane Smith',
    email: 'jane.smith@example.com',
    avatar: 'https://avatars.githubusercontent.com/u/2?v=4',
    membership: 'free',
    articlesRead: 23,
    joinDate: 'Feb 1, 2026',
    lastActive: '1d ago',
    status: 'active'
  },
  {
    id: 3,
    name: 'Bob Johnson',
    email: 'bob.johnson@example.com',
    avatar: 'https://avatars.githubusercontent.com/u/3?v=4',
    membership: 'premium',
    articlesRead: 89,
    joinDate: 'Dec 10, 2025',
    lastActive: '5h ago',
    status: 'active'
  },
  {
    id: 4,
    name: 'Alice Brown',
    email: 'alice.brown@example.com',
    avatar: 'https://avatars.githubusercontent.com/u/4?v=4',
    membership: 'free',
    articlesRead: 12,
    joinDate: 'Feb 20, 2026',
    lastActive: '3d ago',
    status: 'inactive'
  },
  {
    id: 5,
    name: 'Charlie Wilson',
    email: 'charlie.wilson@example.com',
    avatar: 'https://avatars.githubusercontent.com/u/5?v=4',
    membership: 'free',
    articlesRead: 5,
    joinDate: 'Feb 25, 2026',
    lastActive: 'Never',
    status: 'banned'
  }
]

const getActionItems = (user: any) => [
  [{
    label: 'View Profile',
    icon: 'i-lucide-eye'
  }, {
    label: 'Edit User',
    icon: 'i-lucide-edit'
  }],
  [{
    label: 'Send Email',
    icon: 'i-lucide-mail'
  }, {
    label: 'Reset Password',
    icon: 'i-lucide-key'
  }],
  [{
    label: user.status === 'banned' ? 'Unban User' : 'Ban User',
    icon: 'i-lucide-ban',
    color: 'error'
  }, {
    label: 'Delete User',
    icon: 'i-lucide-trash-2',
    color: 'error'
  }]
]
</script>