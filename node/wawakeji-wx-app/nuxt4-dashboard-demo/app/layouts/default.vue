<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-950">
    <!-- Navigation -->
    <header class="sticky top-0 z-50 border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex items-center justify-between h-16">
          <!-- Logo -->
          <NuxtLink to="/" class="flex items-center gap-2">
            <UIcon name="i-lucide-book-open" class="w-8 h-8 text-primary" />
            <span class="text-xl font-bold hidden sm:block">English Reading</span>
          </NuxtLink>

          <!-- Desktop Navigation -->
          <nav class="hidden md:flex items-center gap-6">
            <NuxtLink to="/articles" class="text-gray-600 dark:text-gray-300 hover:text-primary transition">
              Articles
            </NuxtLink>
            <NuxtLink to="/categories/technology" class="text-gray-600 dark:text-gray-300 hover:text-primary transition">
              Categories
            </NuxtLink>
            <NuxtLink to="/membership" class="text-gray-600 dark:text-gray-300 hover:text-primary transition">
              Membership
            </NuxtLink>
          </nav>

          <!-- Right Side -->
          <div class="flex items-center gap-3">
            <!-- Search -->
            <UButton icon="i-lucide-search" color="neutral" variant="ghost" size="sm" class="hidden sm:flex" />

            <!-- Theme Toggle -->
            <UButton
              :icon="colorMode.value === 'dark' ? 'i-lucide-sun' : 'i-lucide-moon'"
              color="neutral"
              variant="ghost"
              size="sm"
              @click="toggleTheme"
            />

            <!-- User Menu (Desktop) -->
            <UDropdownMenu
              :items="userMenuItems"
              class="hidden md:block"
            >
              <UButton icon="i-lucide-user" color="neutral" variant="ghost" size="sm" />
            </UDropdownMenu>

            <!-- Mobile Menu Button -->
            <UButton
              icon="i-lucide-menu"
              color="neutral"
              variant="ghost"
              size="sm"
              class="md:hidden"
              @click="mobileMenuOpen = !mobileMenuOpen"
            />
          </div>
        </div>
      </div>

      <!-- Mobile Navigation -->
      <div v-if="mobileMenuOpen" class="md:hidden border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
        <nav class="px-4 py-3 space-y-2">
          <NuxtLink to="/articles" class="block py-2 text-gray-600 dark:text-gray-300" @click="mobileMenuOpen = false">
            Articles
          </NuxtLink>
          <NuxtLink to="/categories/technology" class="block py-2 text-gray-600 dark:text-gray-300" @click="mobileMenuOpen = false">
            Categories
          </NuxtLink>
          <NuxtLink to="/membership" class="block py-2 text-gray-600 dark:text-gray-300" @click="mobileMenuOpen = false">
            Membership
          </NuxtLink>
          <div class="border-t border-gray-200 dark:border-gray-700 pt-2 mt-2">
            <NuxtLink to="/user" class="block py-2 text-gray-600 dark:text-gray-300" @click="mobileMenuOpen = false">
              Profile
            </NuxtLink>
            <NuxtLink to="/login" class="block py-2 text-gray-600 dark:text-gray-300" @click="mobileMenuOpen = false">
              Login
            </NuxtLink>
          </div>
        </nav>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1">
      <slot />
    </main>

    <!-- Footer -->
    <footer class="border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 mt-auto">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div class="grid grid-cols-1 md:grid-cols-4 gap-8">
          <!-- Brand -->
          <div class="col-span-1 md:col-span-2">
            <div class="flex items-center gap-2 mb-4">
              <UIcon name="i-lucide-book-open" class="w-6 h-6 text-primary" />
              <span class="text-lg font-bold">English Reading</span>
            </div>
            <p class="text-gray-500 dark:text-gray-400 text-sm max-w-md">
              Improve your English reading skills with our curated articles, vocabulary tools, and progress tracking.
            </p>
          </div>

          <!-- Links -->
          <div>
            <h4 class="font-semibold mb-3">Quick Links</h4>
            <ul class="space-y-2 text-sm text-gray-500 dark:text-gray-400">
              <li><NuxtLink to="/articles" class="hover:text-primary">Articles</NuxtLink></li>
              <li><NuxtLink to="/membership" class="hover:text-primary">Membership</NuxtLink></li>
              <li><NuxtLink to="/user/history" class="hover:text-primary">Reading History</NuxtLink></li>
            </ul>
          </div>

          <!-- Legal -->
          <div>
            <h4 class="font-semibold mb-3">Legal</h4>
            <ul class="space-y-2 text-sm text-gray-500 dark:text-gray-400">
              <li><a href="#" class="hover:text-primary">Privacy Policy</a></li>
              <li><a href="#" class="hover:text-primary">Terms of Service</a></li>
              <li><a href="#" class="hover:text-primary">Contact Us</a></li>
            </ul>
          </div>
        </div>

        <div class="border-t border-gray-200 dark:border-gray-800 mt-8 pt-6 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>&copy; 2026 English Reading App. All rights reserved.</p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
const colorMode = useColorMode()
const mobileMenuOpen = ref(false)

const toggleTheme = () => {
  colorMode.preference = colorMode.value === 'dark' ? 'light' : 'dark'
}

const userMenuItems = [
  [{
    label: 'Profile',
    icon: 'i-lucide-user',
    to: '/user'
  }],
  [{
    label: 'Reading History',
    icon: 'i-lucide-history',
    to: '/user/history'
  }, {
    label: 'Bookmarks',
    icon: 'i-lucide-bookmark',
    to: '/user/bookmarks'
  }, {
    label: 'Vocabulary',
    icon: 'i-lucide-book',
    to: '/user/vocabulary'
  }],
  [{
    label: 'Settings',
    icon: 'i-lucide-settings',
    to: '/user/settings'
  }],
  [{
    label: 'Logout',
    icon: 'i-lucide-log-out',
    to: '/login'
  }]
]
</script>