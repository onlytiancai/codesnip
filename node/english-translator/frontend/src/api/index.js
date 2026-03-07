import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

export default {
  // Projects
  async listProjects() {
    const response = await api.get('/projects')
    return response.data
  },

  async getProject(projectId) {
    const response = await api.get(`/projects/${projectId}`)
    return response.data
  },

  async createProject(data) {
    const response = await api.post('/translation/projects', data)
    return response.data
  },

  async deleteProject(projectId) {
    const response = await api.delete(`/projects/${projectId}`)
    return response.data
  },

  // Translation workflow
  async startTranslation(projectId) {
    const response = await api.post(`/translation/projects/${projectId}/start`)
    return response.data
  },

  async resumeTranslation(projectId) {
    const response = await api.post(`/translation/projects/${projectId}/resume`)
    return response.data
  },

  async confirmStep(projectId, step, approved, modifications = null) {
    const response = await api.post('/translation/confirm', {
      project_id: projectId,
      step,
      approved,
      modifications
    })
    return response.data
  },

  async updateTranslation(projectId, filePath, content) {
    const response = await api.post('/translation/update', {
      project_id: projectId,
      file_path: filePath,
      content
    })
    return response.data
  },

  async getFile(projectId, filePath) {
    const response = await api.get(`/translation/projects/${projectId}/files/${filePath}`)
    return response.data
  },

  // WebSocket connection
  connectWebSocket(projectId, onMessage, onError) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/api/translation/${projectId}`)

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      onMessage(data)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      if (onError) onError(error)
    }

    return ws
  }
}