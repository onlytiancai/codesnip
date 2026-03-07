<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useTranslationStore } from '@/stores/translation'

const router = useRouter()
const store = useTranslationStore()

const url = ref('')
const mode = ref('normal')
const targetLanguage = ref('中文')
const isSubmitting = ref(false)

const modes = [
  { value: 'fast', label: 'Fast', description: 'Quick translation without analysis' },
  { value: 'normal', label: 'Normal', description: 'Analysis, terminology, and translation' },
  { value: 'fine', label: 'Fine', description: 'Full workflow with review and revision' }
]

async function handleSubmit() {
  if (!url.value) return

  isSubmitting.value = true
  try {
    const project = await store.createProject(url.value, mode.value, targetLanguage.value)
    router.push(`/project/${project.project_id}`)
  } catch (e) {
    console.error('Failed to create project:', e)
  } finally {
    isSubmitting.value = false
  }
}
</script>

<template>
  <div class="max-w-2xl mx-auto">
    <div class="bg-white shadow rounded-lg p-6">
      <h1 class="text-2xl font-bold text-gray-900 mb-6">New Translation</h1>

      <form @submit.prevent="handleSubmit" class="space-y-6">
        <!-- URL Input -->
        <div>
          <label for="url" class="block text-sm font-medium text-gray-700 mb-2">
            Article URL
          </label>
          <input
            id="url"
            v-model="url"
            type="url"
            placeholder="https://example.com/article"
            class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
            required
          />
        </div>

        <!-- Target Language -->
        <div>
          <label for="language" class="block text-sm font-medium text-gray-700 mb-2">
            Target Language
          </label>
          <input
            id="language"
            v-model="targetLanguage"
            type="text"
            placeholder="中文"
            class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
          />
        </div>

        <!-- Translation Mode -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Translation Mode
          </label>
          <div class="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <label
              v-for="m in modes"
              :key="m.value"
              class="relative flex cursor-pointer rounded-lg border p-4 focus:outline-none"
              :class="mode === m.value ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-500' : 'border-gray-200'"
            >
              <input
                type="radio"
                v-model="mode"
                :value="m.value"
                class="sr-only"
              />
              <div class="flex flex-col">
                <span class="block text-sm font-medium" :class="mode === m.value ? 'text-primary-900' : 'text-gray-900'">
                  {{ m.label }}
                </span>
                <span class="block text-xs mt-1" :class="mode === m.value ? 'text-primary-700' : 'text-gray-500'">
                  {{ m.description }}
                </span>
              </div>
            </label>
          </div>
        </div>

        <!-- Submit Button -->
        <div class="flex justify-end">
          <button
            type="submit"
            :disabled="isSubmitting || !url"
            class="px-6 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {{ isSubmitting ? 'Creating...' : 'Start Translation' }}
          </button>
        </div>
      </form>
    </div>
  </div>
</template>