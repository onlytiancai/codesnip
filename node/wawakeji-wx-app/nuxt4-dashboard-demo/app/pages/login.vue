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

        <form class="space-y-4">
          <UFormField label="Email" name="email" required>
            <UInput placeholder="your@email.com" type="email" icon="i-lucide-mail" />
          </UFormField>

          <UFormField label="Password" name="password" required>
            <UInput placeholder="Enter your password" type="password" icon="i-lucide-lock" />
          </UFormField>

          <div class="flex items-center justify-between">
            <UCheckbox label="Remember me" />
            <NuxtLink to="#" class="text-sm text-primary hover:underline">Forgot password?</NuxtLink>
          </div>

          <UButton type="submit" block size="lg">
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

        <UButton variant="outline" block size="lg" icon="i-lucide-message-circle">
          Continue with WeChat
        </UButton>

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
</script>