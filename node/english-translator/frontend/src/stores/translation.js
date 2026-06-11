import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '@/api'

export const useTranslationStore = defineStore('translation', () => {
  // State
  const projects = ref([])
  const currentProject = ref(null)
  const isLoading = ref(false)
  const error = ref(null)
  const websocket = ref(null)

  // Workflow state
  const workflowStatus = ref({
    currentStep: null,
    stepData: null,
    message: null,
    isWaitingConfirmation: false,
    confirmationStep: null,
    confirmationContent: null,
    confirmationFilePath: null
  })

  // Computed
  const sortedProjects = computed(() => {
    return [...projects.value].sort((a, b) =>
      new Date(b.updated_at) - new Date(a.updated_at)
    )
  })

  // Actions
  async function fetchProjects() {
    isLoading.value = true
    error.value = null
    try {
      projects.value = await api.listProjects()
    } catch (e) {
      error.value = e.message
    } finally {
      isLoading.value = false
    }
  }

  async function fetchProject(projectId) {
    isLoading.value = true
    error.value = null
    try {
      currentProject.value = await api.getProject(projectId)
    } catch (e) {
      error.value = e.message
    } finally {
      isLoading.value = false
    }
  }

  async function createProject(url, mode, targetLanguage) {
    isLoading.value = true
    error.value = null
    try {
      const project = await api.createProject({
        url,
        mode,
        target_language: targetLanguage
      })
      projects.value.unshift(project)
      return project
    } catch (e) {
      error.value = e.message
      throw e
    } finally {
      isLoading.value = false
    }
  }

  async function deleteProject(projectId) {
    try {
      await api.deleteProject(projectId)
      projects.value = projects.value.filter(p => p.project_id !== projectId)
      if (currentProject.value?.project_id === projectId) {
        currentProject.value = null
      }
    } catch (e) {
      error.value = e.message
      throw e
    }
  }

  async function startTranslation(projectId) {
    try {
      await api.startTranslation(projectId)
      await connectWebSocket(projectId)
    } catch (e) {
      error.value = e.message
      throw e
    }
  }

  async function resumeTranslation(projectId) {
    try {
      await api.resumeTranslation(projectId)
      await connectWebSocket(projectId)
    } catch (e) {
      error.value = e.message
      throw e
    }
  }

  async function confirmStep(step, approved, modifications = null) {
    if (!currentProject.value) return

    try {
      await api.confirmStep(
        currentProject.value.project_id,
        step,
        approved,
        modifications
      )

      // Clear confirmation state
      workflowStatus.value.isWaitingConfirmation = false
      workflowStatus.value.confirmationStep = null
      workflowStatus.value.confirmationContent = null
    } catch (e) {
      error.value = e.message
      throw e
    }
  }

  async function updateFile(filePath, content) {
    if (!currentProject.value) return

    try {
      await api.updateTranslation(
        currentProject.value.project_id,
        filePath,
        content
      )
    } catch (e) {
      error.value = e.message
      throw e
    }
  }

  async function getFile(filePath) {
    if (!currentProject.value) return null

    try {
      const result = await api.getFile(currentProject.value.project_id, filePath)
      return result.content
    } catch (e) {
      error.value = e.message
      return null
    }
  }

  function connectWebSocket(projectId) {
    if (websocket.value) {
      websocket.value.close()
    }

    websocket.value = api.connectWebSocket(
      projectId,
      handleMessage,
      handleError
    )
  }

  function handleMessage(data) {
    console.log('WebSocket message:', data)

    switch (data.type) {
      case 'progress':
        workflowStatus.value.currentStep = data.step
        workflowStatus.value.message = data.message
        workflowStatus.value.stepData = data.data
        break

      case 'confirmation':
        workflowStatus.value.isWaitingConfirmation = true
        workflowStatus.value.confirmationStep = data.step
        workflowStatus.value.confirmationContent = data.data?.content
        workflowStatus.value.confirmationFilePath = data.data?.file_path
        break

      case 'error':
        error.value = data.message
        break

      case 'complete':
        workflowStatus.value.message = 'Translation completed'
        // Refresh project data
        fetchProject(projectId)
        break
    }
  }

  function handleError(error) {
    console.error('WebSocket error:', error)
    error.value = 'Connection error'
  }

  function disconnectWebSocket() {
    if (websocket.value) {
      websocket.value.close()
      websocket.value = null
    }
  }

  function resetWorkflowStatus() {
    workflowStatus.value = {
      currentStep: null,
      stepData: null,
      message: null,
      isWaitingConfirmation: false,
      confirmationStep: null,
      confirmationContent: null,
      confirmationFilePath: null
    }
  }

  return {
    // State
    projects,
    currentProject,
    isLoading,
    error,
    workflowStatus,
    // Computed
    sortedProjects,
    // Actions
    fetchProjects,
    fetchProject,
    createProject,
    deleteProject,
    startTranslation,
    resumeTranslation,
    confirmStep,
    updateFile,
    getFile,
    connectWebSocket,
    disconnectWebSocket,
    resetWorkflowStatus
  }
})