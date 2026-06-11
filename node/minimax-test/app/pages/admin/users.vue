<script setup lang="ts">
definePageMeta({
  middleware: ['auth']
})

const { user } = useUserSession()

if (user.value?.id !== '1') {
  navigateTo('/dashboard')
}

const { data, refresh } = await useFetch<{ users: any[] }>('/api/admin/users')

const users = computed(() => data.value?.users || [])

async function disableUser(userId: string) {
  try {
    await $fetch(`/api/admin/users/${userId}/disable`, { method: 'POST' })
    await refresh()
  } catch (e) {
    console.error('Failed to disable user:', e)
  }
}

async function enableUser(userId: string) {
  try {
    await $fetch(`/api/admin/users/${userId}/enable`, { method: 'POST' })
    await refresh()
  } catch (e) {
    console.error('Failed to enable user:', e)
  }
}
</script>

<template>
  <div class="max-w-6xl mx-auto">
    <h1 class="text-2xl font-bold text-gray-900 mb-6">User Management</h1>

    <div class="bg-white shadow rounded-lg overflow-hidden">
      <table class="min-w-full divide-y divide-gray-200">
        <thead class="bg-gray-50">
          <tr>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              User
            </th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Articles
            </th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Status
            </th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Joined
            </th>
            <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
              Actions
            </th>
          </tr>
        </thead>
        <tbody class="bg-white divide-y divide-gray-200">
          <tr v-for="u in users" :key="u.id">
            <td class="px-6 py-4 whitespace-nowrap">
              <div class="flex items-center">
                <div>
                  <div class="text-sm font-medium text-gray-900">
                    {{ u.name || 'No name' }}
                  </div>
                  <div class="text-sm text-gray-500">{{ u.email }}</div>
                </div>
              </div>
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
              <div class="text-sm text-gray-900">{{ u.articleCount }}</div>
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
              <span
                v-if="u.isDisabled"
                class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800"
              >
                Disabled
              </span>
              <span
                v-else
                class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800"
              >
                Active
              </span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
              {{ new Date(u.createdAt).toLocaleDateString() }}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
              <div class="flex justify-end gap-2">
                <NuxtLink
                  :to="`/admin/users/${u.id}`"
                  class="text-blue-500 hover:text-blue-600"
                >
                  View
                </NuxtLink>
                <button
                  v-if="u.id !== '1'"
                  @click="u.isDisabled ? enableUser(u.id) : disableUser(u.id)"
                  :class="[
                    'text-xs px-2 py-1 rounded',
                    u.isDisabled
                      ? 'bg-green-100 text-green-700 hover:bg-green-200'
                      : 'bg-red-100 text-red-700 hover:bg-red-200'
                  ]"
                >
                  {{ u.isDisabled ? 'Enable' : 'Disable' }}
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
