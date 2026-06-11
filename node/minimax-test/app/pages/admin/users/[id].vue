<script setup lang="ts">
definePageMeta({
  middleware: ['auth']
})

const { user } = useUserSession()
const route = useRoute()

if (user.value?.id !== '1') {
  navigateTo('/dashboard')
}

const userId = route.params.id as string

const { data, refresh } = await useFetch<{ user: any; articles: any[] }>(
  `/api/admin/users/${userId}/articles`
)

const selectedUser = computed(() => data.value?.user)
const articles = computed(() => data.value?.articles || [])

const newPassword = ref('')
const showPasswordForm = ref(false)
const passwordError = ref('')
const passwordSuccess = ref(false)

async function resetPassword() {
  passwordError.value = ''
  passwordSuccess.value = false

  if (!newPassword.value || newPassword.value.length < 6) {
    passwordError.value = 'Password must be at least 6 characters'
    return
  }

  try {
    await $fetch(`/api/admin/users/${userId}/reset-password`, {
      method: 'POST',
      body: { newPassword: newPassword.value }
    })
    newPassword.value = ''
    showPasswordForm.value = false
    passwordSuccess.value = true
    setTimeout(() => { passwordSuccess.value = false }, 3000)
  } catch (e: any) {
    passwordError.value = e.data?.message || 'Failed to reset password'
  }
}

async function togglePublish(articleId: string, isPublished: boolean) {
  if (isPublished) {
    try {
      await $fetch(`/api/admin/articles/${articleId}/disable`, { method: 'POST' })
      await refresh()
    } catch (e) {
      console.error('Failed to unpublish:', e)
    }
  }
}
</script>

<template>
  <div class="max-w-6xl mx-auto">
    <div class="mb-6">
      <NuxtLink to="/admin/users" class="text-blue-500 hover:text-blue-600 text-sm">
        &larr; Back to User Management
      </NuxtLink>
    </div>

    <div v-if="selectedUser" class="space-y-6">
      <!-- User Info -->
      <div class="bg-white shadow rounded-lg p-6">
        <div class="flex justify-between items-start">
          <div>
            <h2 class="text-xl font-bold text-gray-900">
              {{ selectedUser.name || 'No name' }}
            </h2>
            <p class="text-gray-500">{{ selectedUser.email }}</p>
            <span
              :class="[
                'inline-block mt-2 px-2 py-1 text-xs font-semibold rounded',
                selectedUser.isDisabled
                  ? 'bg-red-100 text-red-800'
                  : 'bg-green-100 text-green-800'
              ]"
            >
              {{ selectedUser.isDisabled ? 'Disabled' : 'Active' }}
            </span>
          </div>

          <div v-if="selectedUser.id !== '1'" class="space-y-2">
            <button
              v-if="!showPasswordForm"
              @click="showPasswordForm = true"
              class="w-full px-4 py-2 bg-blue-500 text-white text-sm rounded-md hover:bg-blue-600"
            >
              Reset Password
            </button>

            <div v-if="showPasswordForm" class="space-y-2">
              <input
                v-model="newPassword"
                type="password"
                placeholder="New password"
                class="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
              />
              <div v-if="passwordError" class="text-red-500 text-xs">
                {{ passwordError }}
              </div>
              <div class="flex gap-2">
                <button
                  @click="resetPassword"
                  class="flex-1 px-3 py-1 bg-green-500 text-white text-xs rounded-md hover:bg-green-600"
                >
                  Confirm
                </button>
                <button
                  @click="showPasswordForm = false; newPassword = ''"
                  class="flex-1 px-3 py-1 bg-gray-200 text-gray-700 text-xs rounded-md hover:bg-gray-300"
                >
                  Cancel
                </button>
              </div>
            </div>

            <div v-if="passwordSuccess" class="text-green-500 text-xs">
              Password reset successfully
            </div>
          </div>
        </div>
      </div>

      <!-- User Articles -->
      <div class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-6 py-4 border-b border-gray-200">
          <h3 class="text-lg font-medium text-gray-900">
            Articles ({{ articles.length }})
          </h3>
        </div>

        <div v-if="articles.length === 0" class="p-6 text-center text-gray-500">
          No articles yet
        </div>

        <table v-else class="min-w-full divide-y divide-gray-200">
          <thead class="bg-gray-50">
            <tr>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Title
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Created
              </th>
              <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-200">
            <tr v-for="article in articles" :key="article.id">
              <td class="px-6 py-4">
                <NuxtLink
                  :to="`/articles/${article.id}`"
                  class="text-blue-500 hover:text-blue-600"
                >
                  {{ article.title }}
                </NuxtLink>
              </td>
              <td class="px-6 py-4 whitespace-nowrap">
                <span
                  :class="[
                    'px-2 inline-flex text-xs leading-5 font-semibold rounded-full',
                    article.isPublished
                      ? 'bg-green-100 text-green-800'
                      : 'bg-yellow-100 text-yellow-800'
                  ]"
                >
                  {{ article.isPublished ? 'Published' : 'Draft' }}
                </span>
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {{ new Date(article.createdAt).toLocaleDateString() }}
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-right">
                <button
                  v-if="article.isPublished"
                  @click="togglePublish(article.id, true)"
                  class="text-red-500 hover:text-red-600 text-xs"
                >
                  Unpublish
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
