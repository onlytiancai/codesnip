<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-950 flex">
    <!-- Sidebar -->
    <aside
      :class="[
        'fixed inset-y-0 left-0 z-50 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 transform transition-all duration-200 lg:translate-x-0',
        sidebarOpen ? 'translate-x-0' : '-translate-x-full',
        sidebarCollapsed ? 'w-16' : 'w-64'
      ]"
    >
      <!-- Logo -->
      <div class="flex items-center gap-2 h-16 px-4 border-b border-gray-200 dark:border-gray-800">
        <UIcon name="i-lucide-book-open" class="w-8 h-8 text-primary flex-shrink-0" />
        <span v-if="!sidebarCollapsed" class="text-xl font-bold truncate">Admin Panel</span>
      </div>

      <!-- Navigation -->
      <nav class="p-2 space-y-1">
        <NuxtLink
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          :class="[
            'flex items-center gap-3 px-3 py-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition',
            sidebarCollapsed ? 'justify-center' : ''
          ]"
          active-class="bg-primary/10 text-primary font-medium"
          :title="sidebarCollapsed ? item.label : undefined"
        >
          <UIcon :name="item.icon" class="w-5 h-5 flex-shrink-0" />
          <span v-if="!sidebarCollapsed">{{ item.label }}</span>
          <UBadge v-if="item.badge && !sidebarCollapsed" color="primary" variant="subtle" size="xs">
            {{ item.badge }}
          </UBadge>
        </NuxtLink>
      </nav>

      <!-- Collapse Toggle Button (Desktop only) -->
      <button
        class="hidden lg:flex absolute top-20 -right-3 w-6 h-6 items-center justify-center bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-full shadow-sm hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
        @click="sidebarCollapsed = !sidebarCollapsed"
      >
        <UIcon :name="sidebarCollapsed ? 'i-lucide-chevron-right' : 'i-lucide-chevron-left'" class="w-4 h-4 text-gray-500" />
      </button>

      <!-- User Section -->
      <div class="absolute bottom-0 left-0 right-0 p-2 border-t border-gray-200 dark:border-gray-800">
        <div :class="['flex items-center gap-3', sidebarCollapsed ? 'justify-center' : '']">
          <UAvatar src="https://avatars.githubusercontent.com/u/1?v=4" alt="Admin" size="sm" />
          <div v-if="!sidebarCollapsed" class="flex-1 min-w-0">
            <p class="text-sm font-medium truncate">Admin User</p>
            <p class="text-xs text-gray-500 dark:text-gray-400 truncate">admin@example.com</p>
          </div>
          <UButton v-if="!sidebarCollapsed" icon="i-lucide-log-out" color="neutral" variant="ghost" size="xs" @click="handleLogout" />
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
    <div :class="['flex-1 flex flex-col transition-all duration-200', sidebarCollapsed ? 'lg:ml-16' : 'lg:ml-64']">
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
const router = useRouter()
const sidebarOpen = ref(false)
const sidebarCollapsed = ref(false)
const { loggedIn, clear } = useUserSession()
const toast = useToast()

const toggleTheme = () => {
  colorMode.preference = colorMode.value === 'dark' ? 'light' : 'dark'
}

const handleLogout = async () => {
  try {
    await $fetch('/api/auth/logout', { method: 'POST' })
    await clear()
    router.push('/login')
    toast.add({
      title: 'Logged out',
      description: 'You have been successfully logged out.',
      color: 'success'
    })
  } catch (error) {
    toast.add({
      title: 'Error',
      description: 'Failed to logout. Please try again.',
      color: 'error'
    })
  }
}

const navItems = [
  { label: 'Dashboard', icon: 'i-lucide-layout-dashboard', to: '/admin' },
  { label: 'Articles', icon: 'i-lucide-file-text', to: '/admin/articles', badge: '128' },
  { label: 'Categories', icon: 'i-lucide-folder', to: '/admin/categories' },
  { label: 'Tags', icon: 'i-lucide-tag', to: '/admin/tags' },
  { label: 'Dictionary', icon: 'i-lucide-book', to: '/admin/dictionary' },
  { label: 'Users', icon: 'i-lucide-users', to: '/admin/users' },
  { label: 'Orders', icon: 'i-lucide-credit-card', to: '/admin/orders' },
  { label: 'Analytics', icon: 'i-lucide-bar-chart-2', to: '/admin/analytics' }
]

const currentPageTitle = computed(() => {
  const path = route.path
  if (path === '/admin') return 'Dashboard'
  if (path.includes('/admin/articles/create')) return 'Create Article'
  if (path.includes('/admin/articles') && path.includes('/edit')) return 'Edit Article'
  if (path === '/admin/articles') return 'Articles'
  if (path === '/admin/categories') return 'Categories'
  if (path === '/admin/tags') return 'Tags'
  if (path === '/admin/dictionary') return 'Dictionary'
  if (path === '/admin/users') return 'Users'
  if (path === '/admin/orders') return 'Orders'
  if (path === '/admin/analytics') return 'Analytics'
  return 'Admin'
})
</script>