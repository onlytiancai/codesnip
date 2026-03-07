<script setup>
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div class="min-h-screen bg-gray-50">
    <header class="bg-white shadow-sm border-b border-gray-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center h-16">
          <div class="flex items-center">
            <RouterLink to="/" class="flex items-center space-x-2">
              <svg class="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
              </svg>
              <span class="text-xl font-semibold text-gray-900">Translation Agent</span>
            </RouterLink>
          </div>
          <nav class="flex items-center space-x-4">
            <RouterLink to="/" class="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium">
              History
            </RouterLink>
            <RouterLink to="/new" class="bg-primary-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-primary-700">
              New Translation
            </RouterLink>
          </nav>
        </div>
      </div>
    </header>
    <main class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <RouterView />
    </main>
  </div>
</template>