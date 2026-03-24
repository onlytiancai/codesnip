<script setup lang="ts">
const { loggedIn, user, clear } = useUserSession()
const mobileMenuOpen = ref(false)

async function handleLogout() {
  await clear()
  navigateTo('/login')
}

function closeMobileMenu() {
  mobileMenuOpen.value = false
}
</script>

<template>
  <div class="min-h-screen bg-gray-50">
    <header class="bg-white shadow-sm border-b border-gray-200">
      <nav class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16 items-center">
          <div class="flex items-center gap-4 sm:gap-8">
            <NuxtLink to="/" class="text-xl font-bold text-gray-900">
              Article Scraper
            </NuxtLink>
            <div v-if="loggedIn" class="hidden sm:flex gap-6">
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
              <NuxtLink
                v-if="user?.id === '1'"
                to="/admin/users"
                class="text-gray-600 hover:text-gray-900"
                active-class="text-gray-900 font-medium"
              >
                Admin
              </NuxtLink>
            </div>
          </div>
          <div class="hidden sm:flex items-center gap-4">
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
          <!-- Mobile menu button -->
          <button
            @click="mobileMenuOpen = !mobileMenuOpen"
            class="sm:hidden p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100"
          >
            <svg v-if="!mobileMenuOpen" class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
            <svg v-else class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <!-- Mobile menu -->
        <div v-if="mobileMenuOpen" class="sm:hidden pb-4">
          <div v-if="loggedIn" class="flex flex-col gap-2">
            <NuxtLink
              to="/dashboard"
              @click="closeMobileMenu"
              class="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md"
              active-class="text-gray-900 font-medium bg-gray-100"
            >
              Dashboard
            </NuxtLink>
            <NuxtLink
              to="/articles/new"
              @click="closeMobileMenu"
              class="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md"
              active-class="text-gray-900 font-medium bg-gray-100"
            >
              New Article
            </NuxtLink>
            <NuxtLink
              v-if="user?.id === '1'"
              to="/admin/users"
              @click="closeMobileMenu"
              class="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md"
              active-class="text-gray-900 font-medium bg-gray-100"
            >
              Admin
            </NuxtLink>
            <div class="border-t border-gray-200 pt-2 mt-2">
              <span class="block text-sm text-gray-500 px-3 py-2">{{ user?.email }}</span>
              <button
                @click="handleLogout"
                class="text-left text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md w-full"
              >
                Logout
              </button>
            </div>
          </div>
          <div v-else class="flex flex-col gap-2">
            <NuxtLink
              to="/login"
              @click="closeMobileMenu"
              class="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md"
              active-class="text-gray-900 font-medium bg-gray-100"
            >
              Login
            </NuxtLink>
            <NuxtLink
              to="/register"
              @click="closeMobileMenu"
              class="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md"
              active-class="text-gray-900 font-medium bg-gray-100"
            >
              Register
            </NuxtLink>
          </div>
        </div>
      </nav>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <slot />
    </main>
  </div>
</template>
