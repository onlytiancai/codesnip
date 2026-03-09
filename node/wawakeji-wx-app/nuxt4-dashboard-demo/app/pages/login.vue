<template>
  <NuxtLayout name="default">
    <div class="min-h-[80vh] flex items-center justify-center py-12 px-4">
      <UCard class="w-full max-w-md">
        <template #header>
          <div class="text-center">
            <UIcon name="i-lucide-book-open" class="w-12 h-12 text-primary mx-auto mb-4" />
            <h1 class="text-2xl font-bold">Welcome Back</h1>
            <p class="text-gray-500 dark:text-gray-400 mt-2">Sign in to continue your learning journey</p>
          </div>
        </template>

        <form class="space-y-4" @submit.prevent="handleLogin">
          <UFormField label="Email" name="email" required :error="errors.email">
            <UInput v-model="form.email" placeholder="your@email.com" type="email" icon="i-lucide-mail" />
          </UFormField>

          <UFormField label="Password" name="password" required :error="errors.password">
            <UInput v-model="form.password" placeholder="Enter your password" type="password" icon="i-lucide-lock" />
          </UFormField>

          <div class="flex items-center justify-between">
            <UCheckbox v-model="form.remember" label="Remember me" />
            <NuxtLink to="#" class="text-sm text-primary hover:underline">Forgot password?</NuxtLink>
          </div>

          <UButton type="submit" block size="lg" :loading="loading">
            Sign In
          </UButton>
        </form>

        <div class="relative my-6">
          <div class="absolute inset-0 flex items-center">
            <div class="w-full border-t border-gray-200 dark:border-gray-700" />
          </div>
          <div class="relative flex justify-center text-sm">
            <span class="px-2 bg-white dark:bg-gray-900 text-gray-500">Or continue with</span>
          </div>
        </div>

        <div class="space-y-3">
          <UButton variant="outline" block size="lg" icon="i-simple-icons-github" @click="loginWithGitHub">
            Continue with GitHub
          </UButton>
          <UButton variant="outline" block size="lg" icon="i-simple-icons-google" @click="loginWithGoogle">
            Continue with Google
          </UButton>
        </div>

        <template #footer>
          <p class="text-center text-sm text-gray-500 dark:text-gray-400">
            Don't have an account?
            <NuxtLink to="/register" class="text-primary hover:underline font-medium">Sign up</NuxtLink>
          </p>
        </template>
      </UCard>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false
})

const { fetch: fetchSession } = useUserSession()
const router = useRouter()
const toast = useToast()

const form = reactive({
  email: '',
  password: '',
  remember: false
})

const errors = reactive({
  email: '',
  password: ''
})

const loading = ref(false)

const handleLogin = async () => {
  errors.email = ''
  errors.password = ''

  if (!form.email) {
    errors.email = 'Email is required'
    return
  }
  if (!form.password) {
    errors.password = 'Password is required'
    return
  }

  loading.value = true

  try {
    await $fetch('/api/auth/login', {
      method: 'POST',
      body: {
        email: form.email,
        password: form.password
      }
    })

    await fetchSession()
    router.push('/')
  } catch (error: any) {
    const message = error?.data?.message || error?.message || 'Login failed'
    toast.add({
      title: 'Login Failed',
      description: message,
      color: 'error'
    })
  } finally {
    loading.value = false
  }
}

const loginWithGitHub = () => {
  window.location.href = '/auth/github'
}

const loginWithGoogle = () => {
  window.location.href = '/auth/google'
}
</script>