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
          <UButton variant="outline" block size="lg" @click="loginWithWeChat">
            <template #leading>
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="1.2em" height="1.2em">
                <path fill="currentColor" d="M8.691 2.188C3.891 2.188 0 5.476 0 9.53c0 2.212 1.17 4.203 3.002 5.55a.59.59 0 0 1 .213.665l-.39 1.48c-.019.07-.048.141-.048.213 0 .163.13.295.29.295a.326.326 0 0 0 .167-.054l1.903-1.114a.864.864 0 0 1 .717-.098 10.16 10.16 0 0 0 2.837.403c.276 0 .543-.027.811-.05-.857-2.578.157-4.972 1.932-6.446 1.703-1.415 3.882-1.98 5.853-1.838-.576-3.583-4.196-6.348-8.596-6.348zM5.785 5.991c.642 0 1.162.529 1.162 1.18a1.17 1.17 0 0 1-1.162 1.178A1.17 1.17 0 0 1 4.623 7.17c0-.651.52-1.18 1.162-1.18zm5.813 0c.642 0 1.162.529 1.162 1.18a1.17 1.17 0 0 1-1.162 1.178 1.17 1.17 0 0 1-1.162-1.178c0-.651.52-1.18 1.162-1.18zm5.34 2.867c-1.797-.052-3.746.512-5.28 1.786-1.72 1.428-2.687 3.72-1.78 6.22.942 2.453 3.666 4.229 6.884 4.229.826 0 1.622-.12 2.361-.336a.722.722 0 0 1 .598.082l1.584.926a.272.272 0 0 0 .14.047c.134 0 .24-.111.24-.247 0-.06-.023-.12-.038-.177l-.327-1.233a.582.582 0 0 1-.023-.156.49.49 0 0 1 .201-.398C23.024 18.48 24 16.82 24 14.98c0-3.21-2.931-5.837-6.656-6.088V8.89c-.135-.01-.27-.027-.407-.03zm-2.53 3.274c.535 0 .969.44.969.982a.976.976 0 0 1-.969.983.976.976 0 0 1-.969-.983c0-.542.434-.982.97-.982zm4.844 0c.535 0 .969.44.969.982a.976.976 0 0 1-.969.983.976.976 0 0 1-.969-.983c0-.542.434-.982.969-.982z"/>
              </svg>
            </template>
            Continue with WeChat
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

const { fetch: fetchSession, loggedIn } = useUserSession()
const route = useRoute()
const router = useRouter()
const toast = useToast()

// Redirect if already logged in
if (loggedIn.value) {
  router.push('/')
}

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

    // Redirect to the intended page or home
    const redirect = route.query.redirect as string
    router.push(redirect || '/')
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
  // Store redirect path in session storage
  const redirect = route.query.redirect as string
  if (redirect) {
    sessionStorage.setItem('oauth_redirect', redirect)
  }
  window.location.href = '/auth/github'
}

const loginWithGoogle = () => {
  // Store redirect path in session storage
  const redirect = route.query.redirect as string
  if (redirect) {
    sessionStorage.setItem('oauth_redirect', redirect)
  }
  window.location.href = '/auth/google'
}

const loginWithWeChat = () => {
  // Store redirect path in session storage
  const redirect = route.query.redirect as string
  if (redirect) {
    sessionStorage.setItem('oauth_redirect', redirect)
  }

  // Detect if in WeChat browser
  const isWeChat = /MicroMessenger/i.test(navigator.userAgent)

  if (isWeChat) {
    // Use WeChat MP OAuth for in-app browser
    window.location.href = '/auth/wechat-mp'
  } else {
    // Use WeChat Open Platform QR code login for PC browsers
    window.location.href = '/auth/wechat'
  }
}
</script>