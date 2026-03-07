<script setup>
import { onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useTranslationStore } from '@/stores/translation'

const router = useRouter()
const store = useTranslationStore()

onMounted(() => {
  store.fetchProjects()
})

const sortedProjects = computed(() => store.sortedProjects)

function getStatusBadgeClass(status) {
  switch (status) {
    case 'completed':
      return 'bg-green-100 text-green-800'
    case 'in_progress':
      return 'bg-blue-100 text-blue-800'
    case 'waiting_confirmation':
      return 'bg-yellow-100 text-yellow-800'
    case 'error':
      return 'bg-red-100 text-red-800'
    default:
      return 'bg-gray-100 text-gray-800'
  }
}

function formatDate(date) {
  return new Date(date).toLocaleString()
}

function goToProject(projectId) {
  router.push(`/project/${projectId}`)
}

async function handleDelete(projectId, event) {
  event.stopPropagation()
  if (confirm('Are you sure you want to delete this project?')) {
    await store.deleteProject(projectId)
  }
}
</script>

<template>
  <div>
    <div class="flex justify-between items-center mb-6">
      <h1 class="text-2xl font-bold text-gray-900">Translation History</h1>
      <router-link
        to="/new"
        class="bg-primary-600 text-white px-4 py-2 rounded-md hover:bg-primary-700"
      >
        New Translation
      </router-link>
    </div>

    <div v-if="store.isLoading" class="text-center py-12">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
      <p class="mt-4 text-gray-500">Loading projects...</p>
    </div>

    <div v-else-if="sortedProjects.length === 0" class="text-center py-12 bg-white rounded-lg shadow">
      <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <h3 class="mt-2 text-sm font-medium text-gray-900">No translations yet</h3>
      <p class="mt-1 text-sm text-gray-500">Get started by creating a new translation project.</p>
      <div class="mt-6">
        <router-link
          to="/new"
          class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700"
        >
          Create Translation
        </router-link>
      </div>
    </div>

    <div v-else class="bg-white shadow overflow-hidden sm:rounded-lg">
      <ul class="divide-y divide-gray-200">
        <li
          v-for="project in sortedProjects"
          :key="project.project_id"
          @click="goToProject(project.project_id)"
          class="hover:bg-gray-50 cursor-pointer"
        >
          <div class="px-4 py-4 sm:px-6">
            <div class="flex items-center justify-between">
              <div class="flex-1 min-w-0">
                <h3 class="text-sm font-medium text-primary-600 truncate">
                  {{ project.title || project.url }}
                </h3>
                <p class="mt-1 text-sm text-gray-500 truncate">
                  {{ project.url }}
                </p>
              </div>
              <div class="ml-4 flex items-center space-x-2">
                <span
                  :class="getStatusBadgeClass(project.status)"
                  class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                >
                  {{ project.status.replace('_', ' ') }}
                </span>
                <button
                  @click="handleDelete(project.project_id, $event)"
                  class="text-gray-400 hover:text-red-500"
                >
                  <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            </div>
            <div class="mt-2 sm:flex sm:justify-between">
              <div class="sm:flex sm:space-x-4">
                <span class="text-xs text-gray-500">
                  Mode: {{ project.mode }}
                </span>
                <span class="text-xs text-gray-500">
                  Step: {{ project.current_step }}
                </span>
              </div>
              <div class="mt-2 flex items-center text-xs text-gray-500 sm:mt-0">
                <span>Updated {{ formatDate(project.updated_at) }}</span>
              </div>
            </div>
          </div>
        </li>
      </ul>
    </div>
  </div>
</template>