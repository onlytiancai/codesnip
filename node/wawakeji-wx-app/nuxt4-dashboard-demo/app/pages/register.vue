<template>
  <NuxtLayout name="default">
    <div class="min-h-[80vh] flex items-center justify-center py-12 px-4">
      <UCard class="w-full max-w-2xl">
        <template #header>
          <div class="text-center">
            <UIcon name="i-lucide-book-open" class="w-12 h-12 text-primary mx-auto mb-4" />
            <h1 class="text-2xl font-bold">Create Your Account</h1>
            <p class="text-gray-500 dark:text-gray-400 mt-2">Start your English reading journey today</p>
          </div>
        </template>

        <!-- Step Indicator -->
        <div class="flex items-center justify-center gap-2 mb-8">
          <template v-for="step in 3" :key="step">
            <div
              :class="[
                'w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium',
                currentStep >= step
                  ? 'bg-primary text-white'
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-400'
              ]"
            >
              {{ step }}
            </div>
            <div v-if="step < 3" class="w-16 h-0.5 bg-gray-200 dark:bg-gray-700" />
          </template>
        </div>

        <!-- Step 1: Email & Password -->
        <div v-if="currentStep === 1" class="space-y-4">
          <UFormField label="Email" name="email" required :error="errors.email">
            <UInput v-model="form.email" placeholder="your@email.com" type="email" icon="i-lucide-mail" />
          </UFormField>
          <UFormField label="Password" name="password" required :error="errors.password">
            <UInput v-model="form.password" placeholder="Create a password" type="password" icon="i-lucide-lock" />
          </UFormField>
          <UFormField label="Confirm Password" name="confirmPassword" required :error="errors.confirmPassword">
            <UInput v-model="form.confirmPassword" placeholder="Confirm your password" type="password" icon="i-lucide-lock" />
          </UFormField>
        </div>

        <!-- Step 2: Profile Info -->
        <div v-if="currentStep === 2" class="space-y-4">
          <UFormField label="Full Name" name="name" required :error="errors.name">
            <UInput v-model="form.name" placeholder="Your full name" icon="i-lucide-user" />
          </UFormField>
          <UFormField label="English Level" name="level" required>
            <USelect
              v-model="form.level"
              placeholder="Select your level"
              :items="[
                { label: 'Beginner', value: 'beginner' },
                { label: 'Intermediate', value: 'intermediate' },
                { label: 'Advanced', value: 'advanced' }
              ]"
            />
          </UFormField>
          <UFormField label="Daily Reading Goal" name="goal">
            <USelect
              v-model="form.goal"
              placeholder="Select daily goal"
              :items="[
                { label: '5 minutes', value: '5' },
                { label: '10 minutes', value: '10' },
                { label: '15 minutes', value: '15' },
                { label: '30 minutes', value: '30' }
              ]"
            />
          </UFormField>
        </div>

        <!-- Step 3: Interests -->
        <div v-if="currentStep === 3">
          <p class="text-sm text-gray-500 dark:text-gray-400 mb-4">
            Select topics you're interested in (helps us recommend articles)
          </p>
          <div class="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div
              v-for="interest in interests"
              :key="interest.id"
              :class="[
                'p-4 rounded-lg border-2 cursor-pointer transition text-center',
                selectedInterests.includes(interest.id)
                  ? 'border-primary bg-primary/5'
                  : 'border-gray-200 dark:border-gray-700 hover:border-primary/50'
              ]"
              @click="toggleInterest(interest.id)"
            >
              <UIcon :name="interest.icon" class="w-6 h-6 mx-auto mb-2" />
              <span class="text-sm font-medium">{{ interest.name }}</span>
            </div>
          </div>
        </div>

        <div class="flex gap-3 mt-6">
          <UButton
            v-if="currentStep > 1"
            variant="outline"
            @click="currentStep--"
          >
            Back
          </UButton>
          <UButton
            class="flex-1"
            :loading="loading"
            @click="currentStep < 3 ? nextStep() : handleSubmit()"
          >
            {{ currentStep === 3 ? 'Create Account' : 'Continue' }}
          </UButton>
        </div>

        <div class="relative my-6">
          <div class="absolute inset-0 flex items-center">
            <div class="w-full border-t border-gray-200 dark:border-gray-700" />
          </div>
          <div class="relative flex justify-center text-sm">
            <span class="px-2 bg-white dark:bg-gray-900 text-gray-500">Or sign up with</span>
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
            Already have an account?
            <NuxtLink to="/login" class="text-primary hover:underline font-medium">Sign in</NuxtLink>
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

const currentStep = ref(1)
const selectedInterests = ref<string[]>([])
const loading = ref(false)

const form = reactive({
  email: '',
  password: '',
  confirmPassword: '',
  name: '',
  level: '',
  goal: ''
})

const errors = reactive({
  email: '',
  password: '',
  confirmPassword: '',
  name: ''
})

const interests = [
  { id: 'technology', name: 'Technology', icon: 'i-lucide-cpu' },
  { id: 'science', name: 'Science', icon: 'i-lucide-flask-conical' },
  { id: 'business', name: 'Business', icon: 'i-lucide-briefcase' },
  { id: 'health', name: 'Health', icon: 'i-lucide-heart-pulse' },
  { id: 'culture', name: 'Culture', icon: 'i-lucide-globe' },
  { id: 'travel', name: 'Travel', icon: 'i-lucide-plane' },
  { id: 'sports', name: 'Sports', icon: 'i-lucide-trophy' },
  { id: 'entertainment', name: 'Entertainment', icon: 'i-lucide-film' },
  { id: 'education', name: 'Education', icon: 'i-lucide-graduation-cap' }
]

const toggleInterest = (id: string) => {
  const index = selectedInterests.value.indexOf(id)
  if (index === -1) {
    selectedInterests.value.push(id)
  } else {
    selectedInterests.value.splice(index, 1)
  }
}

const validateStep1 = () => {
  errors.email = ''
  errors.password = ''
  errors.confirmPassword = ''

  let valid = true

  if (!form.email) {
    errors.email = 'Email is required'
    valid = false
  } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) {
    errors.email = 'Please enter a valid email'
    valid = false
  }

  if (!form.password) {
    errors.password = 'Password is required'
    valid = false
  } else if (form.password.length < 6) {
    errors.password = 'Password must be at least 6 characters'
    valid = false
  }

  if (form.password !== form.confirmPassword) {
    errors.confirmPassword = 'Passwords do not match'
    valid = false
  }

  return valid
}

const validateStep2 = () => {
  errors.name = ''

  let valid = true

  if (!form.name) {
    errors.name = 'Name is required'
    valid = false
  } else if (form.name.length < 2) {
    errors.name = 'Name must be at least 2 characters'
    valid = false
  }

  return valid
}

const nextStep = () => {
  if (currentStep.value === 1 && validateStep1()) {
    currentStep.value = 2
  } else if (currentStep.value === 2 && validateStep2()) {
    currentStep.value = 3
  }
}

const handleSubmit = async () => {
  if (!validateStep1() || !validateStep2()) {
    return
  }

  loading.value = true

  try {
    await $fetch('/api/auth/register', {
      method: 'POST',
      body: {
        email: form.email,
        password: form.password,
        confirmPassword: form.confirmPassword,
        name: form.name
      }
    })

    await fetchSession()
    router.push('/')
  } catch (error: any) {
    const message = error?.data?.message || error?.message || 'Registration failed'
    toast.add({
      title: 'Registration Failed',
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