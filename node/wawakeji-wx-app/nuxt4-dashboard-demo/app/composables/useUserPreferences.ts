import type { Ref } from 'vue'

interface UserPreferences {
  id: number
  userId: number
  englishLevel: string
  dailyGoal: number
  audioSpeed: number
  theme: string
  fontSize: number
  interests: string[]
  reminderEnabled: boolean
  newArticleNotify: boolean
  vocabReviewNotify: boolean
  marketingEmails: boolean
}

// Global state for preferences
const globalPreferences: Ref<UserPreferences | null> = ref(null)

export const useUserPreferences = () => {
  const preferences = globalPreferences
  const loading = ref(false)
  const error = ref<string | null>(null)
  const colorMode = useColorMode()

  const fetchPreferences = async () => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<UserPreferences>('/api/user/settings')
      preferences.value = response
      // Apply theme from preferences
      if (response.theme) {
        colorMode.preference = response.theme
      }
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch preferences'
      throw e
    } finally {
      loading.value = false
    }
  }

  const updatePreferences = async (data: Partial<{
    englishLevel: string
    dailyGoal: number
    audioSpeed: number
    theme: string
    fontSize: number
    interests: string[]
    reminderEnabled: boolean
    newArticleNotify: boolean
    vocabReviewNotify: boolean
    marketingEmails: boolean
  }>) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<{ preferences: UserPreferences }>('/api/user/settings', {
        method: 'PUT',
        body: data
      })
      preferences.value = response.preferences
      // Apply theme immediately when changed
      if (data.theme) {
        colorMode.preference = data.theme
      }
      return response.preferences
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to update preferences'
      throw e
    } finally {
      loading.value = false
    }
  }

  const changePassword = async (currentPassword: string, newPassword: string) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch('/api/user/password', {
        method: 'PUT',
        body: {
          currentPassword,
          newPassword
        }
      })
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to change password'
      throw e
    } finally {
      loading.value = false
    }
  }

  const deleteAccount = async () => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch('/api/user/account', {
        method: 'DELETE',
        body: {
          confirm: 'DELETE MY ACCOUNT'
        }
      })
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to delete account'
      throw e
    } finally {
      loading.value = false
    }
  }

  return {
    preferences,
    loading,
    error,
    fetchPreferences,
    updatePreferences,
    changePassword,
    deleteAccount
  }
}