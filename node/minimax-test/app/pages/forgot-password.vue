<script setup lang="ts">
definePageMeta({
  layout: 'default'
})

const email = ref('')
const hint = ref<string | null>(null)
const error = ref('')
const loading = ref(false)
const submitted = ref(false)

async function handleSubmit() {
  error.value = ''
  loading.value = true

  try {
    const result = await $fetch<{ hint: string | null }>('/api/auth/password-hint', {
      method: 'POST',
      body: { email: email.value }
    })

    hint.value = result.hint
    submitted.value = true
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : 'Failed to retrieve password hint'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="max-w-md mx-auto mt-12">
    <h1 class="text-2xl font-bold text-gray-900 mb-6">Forgot Password</h1>

    <form @submit.prevent="handleSubmit" class="space-y-4">
      <div v-if="error" class="p-3 bg-red-100 text-red-700 rounded">
        {{ error }}
      </div>

      <template v-if="!submitted">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Email</label>
          <input
            v-model="email"
            type="email"
            required
            class="w-full px-3 py-2 border border-gray-300 rounded-md"
            placeholder="Enter your email"
          />
        </div>

        <button
          type="submit"
          :disabled="loading"
          class="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 disabled:opacity-50"
        >
          {{ loading ? 'Checking...' : 'Get Password Hint' }}
        </button>
      </template>

      <template v-else>
        <div v-if="hint" class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p class="text-sm text-blue-800 font-medium mb-1">Your password hint:</p>
          <p class="text-blue-900">{{ hint }}</p>
        </div>
        <div v-else class="p-4 bg-gray-50 border border-gray-200 rounded-lg">
          <p class="text-gray-600">No password hint was set for this account.</p>
        </div>

        <button
          type="button"
          @click="submitted = false; email = ''; hint = null"
          class="w-full bg-gray-200 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-300"
        >
          Try Another Email
        </button>
      </template>
    </form>

    <p class="mt-4 text-center text-sm text-gray-600">
      Remember your password?
      <NuxtLink to="/login" class="text-blue-500 hover:text-blue-600">
        Login
      </NuxtLink>
    </p>
  </div>
</template>
