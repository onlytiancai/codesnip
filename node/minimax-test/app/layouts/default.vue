<script setup lang="ts">
const { loggedIn, user, clear } = useUserSession()

async function handleLogout() {
  await clear()
  navigateTo('/login')
}
</script>

<template>
  <div class="min-h-screen bg-gray-50">
    <header class="bg-white shadow-sm border-b border-gray-200">
      <nav class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16 items-center">
          <div class="flex items-center gap-8">
            <NuxtLink to="/" class="text-xl font-bold text-gray-900">
              Article Scraper
            </NuxtLink>
            <div v-if="loggedIn" class="flex gap-6">
              <NuxtLink
                to="/dashboard"
                class="text-gray-600 hover:text-gray-900"
                active-class="text-gray-900 font-medium"
              >
                Dashboard
              </NuxtLink>
              <NuxtLink
                to="/articles/new"
                class="text-gray-600 hover:text-gray-900"
                active-class="text-gray-900 font-medium"
              >
                New Article
              </NuxtLink>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <template v-if="loggedIn">
              <span class="text-sm text-gray-600">{{ user?.email }}</span>
              <button
                @click="handleLogout"
                class="text-sm text-gray-600 hover:text-gray-900"
              >
                Logout
              </button>
            </template>
            <template v-else>
              <NuxtLink
                to="/login"
                class="text-sm text-gray-600 hover:text-gray-900"
              >
                Login
              </NuxtLink>
              <NuxtLink
                to="/register"
                class="text-sm text-gray-600 hover:text-gray-900"
              >
                Register
              </NuxtLink>
            </template>
          </div>
        </div>
      </nav>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <slot />
    </main>
  </div>
</template>
