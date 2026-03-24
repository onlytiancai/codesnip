<script setup lang="ts">
definePageMeta({
  layout: 'default'
})

const { loggedIn, fetch: fetchSession } = useUserSession()

if (loggedIn.value) {
  navigateTo('/dashboard')
}

const email = ref('')
const password = ref('')
const error = ref('')
const loading = ref(false)

async function handleLogin() {
  error.value = ''
  loading.value = true

  try {
    await $fetch('/api/auth/login', {
      method: 'POST',
      body: {
        email: email.value,
        password: password.value
      }
    })

    await fetchSession()
    navigateTo('/dashboard')
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : 'Login failed'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="max-w-md mx-auto mt-12">
    <h1 class="text-2xl font-bold text-gray-900 mb-6">Login</h1>

    <form @submit.prevent="handleLogin" class="space-y-4">
      <div v-if="error" class="p-3 bg-red-100 text-red-700 rounded">
        {{ error }}
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">Email</label>
        <input
          v-model="email"
          type="email"
          required
          class="w-full px-3 py-2 border border-gray-300 rounded-md"
        />
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">Password</label>
        <input
          v-model="password"
          type="password"
          required
          class="w-full px-3 py-2 border border-gray-300 rounded-md"
        />
      </div>

      <button
        type="submit"
        :disabled="loading"
        class="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 disabled:opacity-50"
      >
        {{ loading ? 'Logging in...' : 'Login' }}
      </button>
    </form>

    <p class="mt-4 text-center text-sm text-gray-600">
      Don't have an account?
      <NuxtLink to="/register" class="text-blue-500 hover:text-blue-600">
        Register
      </NuxtLink>
    </p>
  </div>
</template>
