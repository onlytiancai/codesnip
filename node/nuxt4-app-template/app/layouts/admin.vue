<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-950 flex">
    <!-- Sidebar -->
    <aside
      :class="[
        'fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 transform transition-transform duration-200 lg:translate-x-0',
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      ]"
    >
      <!-- Logo -->
      <div class="flex items-center gap-2 h-16 px-6 border-b border-gray-200 dark:border-gray-800">
        <UIcon name="i-lucide-book-open" class="w-8 h-8 text-primary" />
        <span class="text-xl font-bold">Admin Panel</span>
      </div>

      <!-- Navigation -->
      <nav class="p-4 space-y-1">
        <NuxtLink
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          class="flex items-center gap-3 px-3 py-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
          active-class="bg-primary/10 text-primary font-medium"
        >
          <UIcon :name="item.icon" class="w-5 h-5" />
          <span>{{ item.label }}</span>
          <UBadge v-if="item.badge" color="primary" variant="subtle" size="xs">
            {{ item.badge }}
          </UBadge>
        </NuxtLink>
      </nav>

      <!-- User Section -->
      <div class="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200 dark:border-gray-800">
        <div class="flex items-center gap-3">
          <UAvatar src="https://avatars.githubusercontent.com/u/1?v=4" alt="Admin" size="sm" />
          <div class="flex-1 min-w-0">
            <p class="text-sm font-medium truncate">Admin User</p>
            <p class="text-xs text-gray-500 dark:text-gray-400 truncate">admin@example.com</p>
          </div>
          <UButton icon="i-lucide-log-out" color="neutral" variant="ghost" size="xs" />
        </div>
      </div>
    </aside>

    <!-- Overlay for mobile -->
    <div
      v-if="sidebarOpen"
      class="fixed inset-0 bg-black/50 z-40 lg:hidden"
      @click="sidebarOpen = false"
    />

    <!-- Main Content -->
    <div class="flex-1 lg:ml-64 flex flex-col">
      <!-- Header -->
      <header class="sticky top-0 z-30 h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
        <div class="flex items-center justify-between h-full px-4 sm:px-6">
          <!-- Left -->
          <div class="flex items-center gap-4">
            <UButton
              icon="i-lucide-menu"
              color="neutral"
              variant="ghost"
              size="sm"
              class="lg:hidden"
              @click="sidebarOpen = !sidebarOpen"
            />
            <!-- Breadcrumb -->
            <div class="hidden sm:flex items-center gap-2 text-sm">
              <span class="text-gray-500 dark:text-gray-400">Admin</span>
              <UIcon name="i-lucide-chevron-right" class="w-4 h-4 text-gray-400" />
              <span class="font-medium">{{ currentPageTitle }}</span>
            </div>
          </div>

          <!-- Right -->
          <div class="flex items-center gap-3">
            <UButton icon="i-lucide-bell" color="neutral" variant="ghost" size="sm" />
            <UButton
              :icon="colorMode.value === 'dark' ? 'i-lucide-sun' : 'i-lucide-moon'"
              color="neutral"
              variant="ghost"
              size="sm"
              @click="toggleTheme"
            />
          </div>
        </div>
      </header>

      <!-- Page Content -->
      <main class="flex-1 p-4 sm:p-6">
        <slot />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
const colorMode = useColorMode()
const route = useRoute()
const sidebarOpen = ref(false)

const toggleTheme = () => {
  colorMode.preference = colorMode.value === 'dark' ? 'light' : 'dark'
}

const navItems = [
  { label: 'Dashboard', icon: 'i-lucide-layout-dashboard', to: '/' },
  { label: 'Articles', icon: 'i-lucide-file-text', to: '/articles', badge: '128' },
  { label: 'Categories', icon: 'i-lucide-folder', to: '/categories' },
  { label: 'Tags', icon: 'i-lucide-tag', to: '/tags' },
  { label: 'Users', icon: 'i-lucide-users', to: '/users' },
  { label: 'Analytics', icon: 'i-lucide-bar-chart-2', to: '/analytics' }
]

const currentPageTitle = computed(() => {
  const path = route.path
  if (path === '/') return 'Dashboard'
  return 'Admin'
})
</script>
