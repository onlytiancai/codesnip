<script setup lang="ts">
definePageMeta({
  layout: 'default'
})

const { loggedIn, fetch: fetchSession } = useUserSession()

if (loggedIn.value) {
  navigateTo('/dashboard')
}

const config = useRuntimeConfig()
const captchaEnabled = computed(() => config.public.captchaEnabled === 'true')

const email = ref('')
const password = ref('')
const captchaAnswer = ref('')
const captchaQuestion = ref('')
const error = ref('')
const loading = ref(false)

// Fetch CAPTCHA if enabled
async function fetchCaptcha() {
  if (!captchaEnabled.value) return
  try {
    const result = await $fetch<{ question: string }>('/api/auth/captcha')
    captchaQuestion.value = result.question
  } catch (e) {
    console.error('Failed to fetch CAPTCHA:', e)
  }
}

onMounted(() => {
  fetchCaptcha()
})

async function handleLogin() {
  error.value = ''
  loading.value = true

  try {
    await $fetch('/api/auth/login', {
      method: 'POST',
      body: {
        email: email.value,
        password: password.value,
        captchaAnswer: captchaAnswer.value || undefined
      }
    })

    await fetchSession()
    await navigateTo('/dashboard')
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : 'Login failed'
    if (captchaEnabled.value) {
      captchaAnswer.value = ''
      fetchCaptcha()
    }
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

      <div v-if="captchaEnabled">
        <label class="block text-sm font-medium text-gray-700 mb-1">
          CAPTCHA: {{ captchaQuestion }}
        </label>
        <input
          v-model="captchaAnswer"
          type="text"
          required
          class="w-full px-3 py-2 border border-gray-300 rounded-md"
          placeholder="Enter the answer"
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

    <div class="mt-4 flex justify-between text-sm">
      <NuxtLink to="/forgot-password" class="text-blue-500 hover:text-blue-600">
        Forgot Password?
      </NuxtLink>
      <span class="text-gray-600">
        Don't have an account?
        <NuxtLink to="/register" class="text-blue-500 hover:text-blue-600">
          Register
        </NuxtLink>
      </span>
    </div>
  </div>
</template>
