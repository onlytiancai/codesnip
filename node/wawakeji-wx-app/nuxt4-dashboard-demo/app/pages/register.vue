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
          <UFormField label="Email" name="email" required>
            <UInput placeholder="your@email.com" type="email" icon="i-lucide-mail" />
          </UFormField>
          <UFormField label="Password" name="password" required>
            <UInput placeholder="Create a password" type="password" icon="i-lucide-lock" />
          </UFormField>
          <UFormField label="Confirm Password" name="confirmPassword" required>
            <UInput placeholder="Confirm your password" type="password" icon="i-lucide-lock" />
          </UFormField>
        </div>

        <!-- Step 2: Profile Info -->
        <div v-if="currentStep === 2" class="space-y-4">
          <UFormField label="Full Name" name="name" required>
            <UInput placeholder="Your full name" icon="i-lucide-user" />
          </UFormField>
          <UFormField label="English Level" name="level" required>
            <USelect
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
            @click="currentStep < 3 ? currentStep++ : handleSubmit()"
          >
            {{ currentStep === 3 ? 'Create Account' : 'Continue' }}
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

const currentStep = ref(1)
const selectedInterests = ref<string[]>([])

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

const handleSubmit = () => {
  // Handle registration
  console.log('Register with interests:', selectedInterests.value)
}
</script>