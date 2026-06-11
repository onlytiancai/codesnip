<script setup>
import { ref, onMounted } from 'vue'

const settings = ref({
  provider: 'openai',
  model: 'gpt-4o',
  apiKey: '',
  apiBase: ''
})

const saved = ref(false)
const loading = ref(false)

onMounted(() => {
  // Load settings from localStorage
  const savedSettings = localStorage.getItem('llm_settings')
  if (savedSettings) {
    settings.value = JSON.parse(savedSettings)
  }
})

async function handleSave() {
  loading.value = true
  try {
    // Save to localStorage
    localStorage.setItem('llm_settings', JSON.stringify(settings.value))
    saved.value = true
    setTimeout(() => {
      saved.value = false
    }, 2000)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="max-w-2xl mx-auto">
    <div class="bg-white shadow rounded-lg p-6">
      <h1 class="text-2xl font-bold text-gray-900 mb-2">Settings</h1>
      <p class="text-sm text-gray-500 mb-6">
        Configure LLM settings in the backend <code class="bg-gray-100 px-1 rounded">.env</code> file for server-side changes.
        Settings below are stored locally in your browser.
      </p>

      <form @submit.prevent="handleSave" class="space-y-6">
        <!-- LLM Provider -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            LLM Provider
          </label>
          <select
            v-model="settings.provider"
            class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
          >
            <option value="openai">OpenAI</option>
            <option value="anthropic">Anthropic</option>
          </select>
        </div>

        <!-- Model -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Model
          </label>
          <select
            v-model="settings.model"
            class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
          >
            <template v-if="settings.provider === 'openai'">
              <option value="gpt-4o">GPT-4o</option>
              <option value="gpt-4o-mini">GPT-4o Mini</option>
              <option value="gpt-4-turbo">GPT-4 Turbo</option>
            </template>
            <template v-else>
              <option value="claude-sonnet-4-6">Claude Sonnet 4.6</option>
              <option value="claude-opus-4-6">Claude Opus 4.6</option>
              <option value="claude-haiku-4-5">Claude Haiku 4.5</option>
            </template>
          </select>
        </div>

        <!-- API Key -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            API Key
          </label>
          <input
            v-model="settings.apiKey"
            type="password"
            placeholder="Enter your API key"
            class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
          />
          <p class="mt-1 text-xs text-gray-500">
            Your API key is stored locally in your browser.
          </p>
        </div>

        <!-- Custom API URL -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Custom API URL (Optional)
          </label>
          <input
            v-model="settings.apiBase"
            type="url"
            placeholder="e.g., http://localhost:8000/v1"
            class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
          />
          <p class="mt-1 text-xs text-gray-500">
            Leave empty to use default API endpoint. Configure in backend <code class="bg-gray-100 px-1 rounded">.env</code> for server-side changes.
          </p>
        </div>

        <!-- Submit Button -->
        <div class="flex justify-end">
          <button
            type="submit"
            :disabled="loading"
            class="px-6 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50"
          >
            {{ saved ? 'Saved!' : 'Save Settings' }}
          </button>
        </div>
      </form>
    </div>

    <!-- Backend Configuration Info -->
    <div class="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
      <h3 class="text-sm font-medium text-blue-800 mb-2">Backend Configuration</h3>
      <p class="text-sm text-blue-700 mb-3">
        To configure LLM settings on the server, edit the <code class="bg-blue-100 px-1 rounded">backend/.env</code> file:
      </p>
      <pre class="text-xs bg-blue-100 p-3 rounded overflow-x-auto text-blue-900"># LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o

# Custom API URL (optional)
OPENAI_API_BASE=http://localhost:8000/v1</pre>
    </div>
  </div>
</template>