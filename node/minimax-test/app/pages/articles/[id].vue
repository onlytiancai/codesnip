<script setup lang="ts">
import TurndownService from 'turndown'

const route = useRoute()
const { loggedIn, user } = useUserSession()

const { data, error, refresh } = await useFetch(`/api/articles/${route.params.id}`)

if (error.value || !data.value) {
  throw createError({
    statusCode: 404,
    message: 'Article not found'
  })
}

const article = computed(() => data.value?.article)
const isOwner = computed(() => loggedIn.value && user.value?.id === article.value?.userId)
const canView = computed(() => article.value?.isPublished || isOwner.value)

if (!canView.value) {
  throw createError({
    statusCode: 403,
    message: 'Access denied'
  })
}

const isEditing = ref(false)
const editedContent = ref('')
const editedTitle = ref('')
const activeTab = ref<'markdown' | 'preview'>('markdown')
const saving = ref(false)
const editTheme = ref<'light' | 'dark' | 'sepia'>('light')
const htmlEditingEnabled = ref(false)
const turndown = new TurndownService()

function startEditing() {
  editedContent.value = article.value?.content || ''
  editedTitle.value = article.value?.title || ''
  isEditing.value = true
}

function cancelEditing() {
  isEditing.value = false
  editedContent.value = ''
  editedTitle.value = ''
  htmlEditingEnabled.value = false
}

function syncHtmlToMarkdown() {
  const html = editedContent.value
  editedContent.value = turndown.turndown(html)
  activeTab.value = 'markdown'
}

async function saveContent() {
  if (!article.value) return

  saving.value = true
  try {
    await $fetch(`/api/articles/${article.value.id}`, {
      method: 'PUT',
      body: {
        title: editedTitle.value,
        content: editedContent.value
      }
    })
    await refresh()
    isEditing.value = false
    htmlEditingEnabled.value = false
  } catch (e) {
    console.error('Failed to save:', e)
    alert('Failed to save changes')
  } finally {
    saving.value = false
  }
}

async function togglePublish() {
  try {
    await $fetch(`/api/articles/${article.value?.id}/publish`, { method: 'POST' })
    await refresh()
  } catch (e) {
    console.error('Failed to toggle publish:', e)
  }
}

function handleHtmlUpdate(content: string) {
  editedContent.value = content
}
</script>

<template>
  <div>
    <div class="mb-6">
      <NuxtLink to="/dashboard" class="text-blue-500 hover:text-blue-600 text-sm">
        &larr; Back to Dashboard
      </NuxtLink>
    </div>

    <article class="max-w-4xl mx-auto">
      <header class="mb-8">
        <div class="flex justify-between items-start">
          <h1 v-if="!isEditing" class="text-3xl font-bold text-gray-900 mb-2">{{ article?.title }}</h1>
          <input
            v-else
            v-model="editedTitle"
            type="text"
            class="text-3xl font-bold text-gray-900 mb-2 w-full px-3 py-2 border border-gray-300 rounded-md"
          />
          <div v-if="isOwner && !isEditing" class="flex gap-2">
            <button
              @click="startEditing"
              class="px-4 py-2 rounded-md text-sm bg-blue-500 text-white hover:bg-blue-600"
            >
              Edit
            </button>
            <button
              @click="togglePublish"
              :class="[
                'px-4 py-2 rounded-md text-sm',
                article?.isPublished
                  ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  : 'bg-green-500 text-white hover:bg-green-600'
              ]"
            >
              {{ article?.isPublished ? 'Unpublish' : 'Publish' }}
            </button>
          </div>
        </div>

        <div class="flex items-center gap-4 text-sm text-gray-500">
          <span v-if="article?.category">{{ article.category.name }}</span>
          <div v-if="article?.tags?.length" class="flex gap-2">
            <span
              v-for="tag in article.tags"
              :key="tag.id"
              class="px-2 py-1 bg-gray-100 rounded-full text-xs"
            >
              {{ tag.name }}
            </span>
          </div>
          <span>{{ new Date(article?.createdAt || '').toLocaleDateString() }}</span>
        </div>

        <p v-if="article?.description && !isEditing" class="mt-4 text-gray-600">
          {{ article.description }}
        </p>
      </header>

      <div v-if="!article?.isPublished && isOwner && !isEditing" class="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <p class="text-yellow-800 text-sm">
          This article is not published. Only you can see it.
        </p>
      </div>

      <div v-if="!isEditing" class="flex items-center justify-between mb-4">
        <ThemeSelector />
      </div>

      <!-- Edit Mode -->
      <div v-if="isEditing" class="space-y-4">
        <div class="flex items-center justify-between border-b border-gray-200 pb-4">
          <div class="flex gap-2">
            <button
              @click="activeTab = 'markdown'"
              :class="[
                'px-4 py-2 text-sm rounded-t-md',
                activeTab === 'markdown'
                  ? 'bg-white border border-b-0 border-gray-200 -mb-px text-blue-600 font-medium'
                  : 'bg-gray-100 text-gray-600 hover:text-gray-800'
              ]"
            >
              Markdown
            </button>
            <button
              @click="activeTab = 'preview'"
              :class="[
                'px-4 py-2 text-sm rounded-t-md flex items-center gap-2',
                activeTab === 'preview'
                  ? 'bg-white border border-b-0 border-gray-200 -mb-px text-blue-600 font-medium'
                  : 'bg-gray-100 text-gray-600 hover:text-gray-800'
              ]"
            >
              Preview
              <select
                v-model="editTheme"
                class="text-xs border border-gray-300 rounded px-1 py-0.5"
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="sepia">Sepia</option>
              </select>
            </button>
          </div>
          <div class="flex gap-2 items-center">
            <label v-if="activeTab === 'preview'" class="flex items-center gap-1 text-xs text-gray-600">
              <input
                v-model="htmlEditingEnabled"
                type="checkbox"
                class="w-3 h-3"
              />
              Enable HTML editing
            </label>
            <button
              v-if="activeTab === 'preview' && htmlEditingEnabled"
              @click="syncHtmlToMarkdown"
              class="px-3 py-1 text-xs bg-purple-500 text-white rounded-md hover:bg-purple-600"
            >
              Sync to Markdown
            </button>
            <button
              @click="cancelEditing"
              class="px-4 py-2 rounded-md text-sm bg-gray-200 text-gray-700 hover:bg-gray-300"
            >
              Cancel
            </button>
            <button
              @click="saveContent"
              :disabled="saving"
              class="px-4 py-2 rounded-md text-sm bg-green-500 text-white hover:bg-green-600 disabled:opacity-50"
            >
              {{ saving ? 'Saving...' : 'Save' }}
            </button>
          </div>
        </div>

        <div class="h-[600px]">
          <textarea
            v-if="activeTab === 'markdown'"
            v-model="editedContent"
            class="w-full h-full px-3 py-2 border border-gray-300 rounded-md font-mono text-sm resize-none"
            placeholder="Enter markdown content..."
          />
          <div v-else class="w-full h-full border border-gray-300 rounded-md overflow-hidden">
            <MarkdownRenderer
              :content="editedContent"
              :theme="editTheme"
              :editable="htmlEditingEnabled"
              @update:content="handleHtmlUpdate"
            />
          </div>
        </div>
      </div>

      <!-- View Mode -->
      <div v-else class="mt-8">
        <MarkdownRenderer :content="article?.content || ''" />
      </div>
    </article>
  </div>
</template>
