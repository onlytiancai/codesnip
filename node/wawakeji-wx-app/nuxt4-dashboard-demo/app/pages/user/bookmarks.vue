<template>
  <NuxtLayout name="default">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Header -->
      <div class="flex items-center justify-between mb-8">
        <div>
          <h1 class="text-2xl font-bold">Bookmarks</h1>
          <p class="text-gray-500 dark:text-gray-400">Articles you've saved for later</p>
        </div>
        <div class="flex items-center gap-2">
          <UInput
            placeholder="Search bookmarks..."
            icon="i-lucide-search"
            size="sm"
            class="w-48"
          />
        </div>
      </div>

      <!-- Premium Banner (for non-premium users) -->
      <UCard class="mb-8 bg-gradient-to-r from-purple-500 to-pink-500 text-white">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-4">
            <UIcon name="i-lucide-crown" class="w-8 h-8" />
            <div>
              <h3 class="font-semibold">Premium Feature</h3>
              <p class="text-sm opacity-90">Unlimited bookmarks with Premium membership</p>
            </div>
          </div>
          <UButton color="white" variant="soft" to="/membership">
            Upgrade
          </UButton>
        </div>
      </UCard>

      <!-- Filter Tabs -->
      <UTabs :items="tabs" class="mb-6">
        <template #all>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <NuxtLink
              v-for="bookmark in bookmarks"
              :key="bookmark.id"
              :to="`/articles/${bookmark.id}`"
            >
              <UCard class="group hover:border-primary transition cursor-pointer h-full">
                <div class="flex gap-4">
                  <img
                    :src="bookmark.cover"
                    :alt="bookmark.title"
                    class="w-24 h-24 object-cover rounded-lg flex-shrink-0"
                  />
                  <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 mb-1">
                      <UBadge color="primary" variant="subtle" size="xs">{{ bookmark.category }}</UBadge>
                    </div>
                    <h4 class="font-medium line-clamp-2 group-hover:text-primary transition">
                      {{ bookmark.title }}
                    </h4>
                    <div class="flex items-center justify-between mt-2">
                      <span class="text-xs text-gray-500 dark:text-gray-400">
                        {{ bookmark.readTime }} min read
                      </span>
                      <UButton
                        icon="i-lucide-bookmark"
                        size="xs"
                        variant="ghost"
                        color="primary"
                      />
                    </div>
                  </div>
                </div>
              </UCard>
            </NuxtLink>
          </div>
        </template>

        <template #collections>
          <div class="mt-4">
            <div class="flex items-center justify-between mb-4">
              <p class="text-sm text-gray-500 dark:text-gray-400">
                Organize your bookmarks into collections
              </p>
              <UButton size="sm" variant="outline" icon="i-lucide-plus">
                New Collection
              </UButton>
            </div>
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <UCard
                v-for="collection in collections"
                :key="collection.id"
                class="hover:border-primary transition cursor-pointer"
              >
                <div class="text-center py-4">
                  <UIcon :name="collection.icon" class="w-10 h-10 text-primary mb-2" />
                  <h4 class="font-medium">{{ collection.name }}</h4>
                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    {{ collection.count }} articles
                  </p>
                </div>
              </UCard>
            </div>
          </div>
        </template>
      </UTabs>

      <!-- Empty State -->
      <div v-if="bookmarks.length === 0" class="text-center py-12">
        <UIcon name="i-lucide-bookmark" class="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
        <h3 class="text-lg font-medium mb-2">No bookmarks yet</h3>
        <p class="text-gray-500 dark:text-gray-400 mb-4">
          Start saving articles to read them later
        </p>
        <UButton to="/articles">Browse Articles</UButton>
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const tabs = [
  { label: 'All Bookmarks', slot: 'all' },
  { label: 'Collections', slot: 'collections' }
]

const bookmarks = [
  {
    id: 1,
    title: 'The Future of Artificial Intelligence in Healthcare',
    category: 'Technology',
    readTime: 8,
    cover: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=200&h=200&fit=crop'
  },
  {
    id: 2,
    title: 'Climate Change: What Scientists Are Saying',
    category: 'Science',
    readTime: 12,
    cover: 'https://images.unsplash.com/photo-1569163139599-0f4517e36f51?w=200&h=200&fit=crop'
  },
  {
    id: 3,
    title: 'Building a Successful Startup: Lessons from Founders',
    category: 'Business',
    readTime: 6,
    cover: 'https://images.unsplash.com/photo-1559136555-9303baea8ebd?w=200&h=200&fit=crop'
  },
  {
    id: 4,
    title: 'The Science of Sleep: Why It Matters',
    category: 'Health',
    readTime: 5,
    cover: 'https://images.unsplash.com/photo-1541781774459-bb2af2f05b55?w=200&h=200&fit=crop'
  }
]

const collections = [
  { id: 1, name: 'Technology', icon: 'i-lucide-cpu', count: 12 },
  { id: 2, name: 'For Work', icon: 'i-lucide-briefcase', count: 8 },
  { id: 3, name: 'Science', icon: 'i-lucide-flask-conical', count: 5 }
]
</script>