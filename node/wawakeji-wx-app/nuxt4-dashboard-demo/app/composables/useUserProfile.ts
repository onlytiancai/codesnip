import type { Ref } from 'vue'

interface UserProfile {
  id: number
  email: string
  name: string | null
  avatar: string | null
  bio: string | null
  role: string
  createdAt: string
}

interface UserStats {
  articlesRead: number
  vocabularyLearned: number
  bookmarks: number
  streak: number
  totalReadingMinutes: number
}

interface Membership {
  plan: string
  startDate?: string
  endDate?: string | null
}

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

interface ProfileResponse {
  user: UserProfile
  stats: UserStats
  membership: Membership
  preferences: UserPreferences | null
}

export const useUserProfile = () => {
  const profile: Ref<ProfileResponse | null> = ref(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  const fetchProfile = async () => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<ProfileResponse>('/api/user/profile')
      profile.value = response
      return response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch profile'
      throw e
    } finally {
      loading.value = false
    }
  }

  const updateProfile = async (data: { name?: string; avatar?: string | null; bio?: string | null }) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<{ user: UserProfile }>('/api/user/profile', {
        method: 'PUT',
        body: data
      })
      if (profile.value) {
        profile.value.user = response.user
      }
      return response.user
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to update profile'
      throw e
    } finally {
      loading.value = false
    }
  }

  return {
    profile,
    loading,
    error,
    fetchProfile,
    updateProfile
  }
}