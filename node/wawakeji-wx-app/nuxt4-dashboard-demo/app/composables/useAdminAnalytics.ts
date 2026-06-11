import type { Ref } from 'vue'

interface Stats {
  totalUsers: number
  totalArticles: number
  publishedArticles: number
  totalCategories: number
  totalTags: number
  totalViews: number
  totalBookmarks: number
  newUsersLast30Days: number
  newArticlesLast30Days: number
}

interface RecentArticle {
  id: number
  title: string
  slug: string
  excerpt: string | null
  status: string
  category: { id: number; name: string } | null
  author: { id: number; name: string | null }
  createdAt: string
}

interface RecentUser {
  id: number
  name: string | null
  email: string
  avatar: string | null
  createdAt: string
}

interface ChartData {
  userRegistrations: { date: string; count: number }[]
  articleCreations: { date: string; count: number }[]
  categoryDistribution: { name: string; count: number }[]
  difficultyDistribution: { difficulty: string; count: number }[]
  topArticles: any[]
  tagUsage: any[]
}

export const useAdminAnalytics = () => {
  const stats: Ref<Stats | null> = ref(null)
  const recentArticles: Ref<RecentArticle[]> = ref([])
  const recentUsers: Ref<RecentUser[]> = ref([])
  const chartData: Ref<ChartData | null> = ref(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Fetch overview data
  const fetchOverview = async () => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<{
        stats: Stats
        recentArticles: RecentArticle[]
        recentUsers: RecentUser[]
      }>('/api/admin/analytics/overview')

      stats.value = response.stats
      recentArticles.value = response.recentArticles
      recentUsers.value = response.recentUsers
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch analytics overview'
      throw e
    } finally {
      loading.value = false
    }
  }

  // Fetch chart data
  const fetchCharts = async (days: number = 30) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<ChartData>(`/api/admin/analytics/charts?days=${days}`)
      chartData.value = response
    } catch (e: any) {
      error.value = e.data?.message || 'Failed to fetch analytics charts'
      throw e
    } finally {
      loading.value = false
    }
  }

  return {
    stats,
    recentArticles,
    recentUsers,
    chartData,
    loading,
    error,
    fetchOverview,
    fetchCharts
  }
}