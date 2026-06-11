import { describe, it, expect, beforeEach } from 'vitest'
import {
  apiRequest,
  loginAsAdmin,
  createTestCategory,
  createTestTag,
  createTestArticle,
  createTestUser,
  cleanupDatabase,
  prisma,
} from '../../setup-e2e'

describe('Admin Analytics API', () => {
  beforeEach(async () => {
    await cleanupDatabase()
    await loginAsAdmin()
  })

  describe('GET /api/admin/analytics/overview', () => {
    it('should return overview stats', async () => {
      const response = await apiRequest('/api/admin/analytics/overview')

      expect(response.status).toBe(200)
      expect(response.data.stats).toBeDefined()
      expect(response.data.stats).toHaveProperty('totalUsers')
      expect(response.data.stats).toHaveProperty('totalArticles')
      expect(response.data.stats).toHaveProperty('publishedArticles')
      expect(response.data.stats).toHaveProperty('totalCategories')
      expect(response.data.stats).toHaveProperty('totalTags')
      expect(response.data.stats).toHaveProperty('totalViews')
      expect(response.data.stats).toHaveProperty('totalBookmarks')
      expect(response.data.stats).toHaveProperty('newUsersLast30Days')
      expect(response.data.stats).toHaveProperty('newArticlesLast30Days')
      expect(response.data).toHaveProperty('recentArticles')
      expect(response.data).toHaveProperty('recentUsers')
    })

    it('should return correct counts', async () => {
      // Create some test data
      const category = await createTestCategory({ name: 'Analytics Cat', slug: 'analytics-cat' })
      const tag = await createTestTag({ name: 'Analytics Tag', slug: 'analytics-tag' })

      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })

      await createTestArticle(admin!.id, {
        status: 'published',
        views: 100,
        bookmarks: 10,
        slug: 'analytics-article',
      })

      const response = await apiRequest('/api/admin/analytics/overview')

      expect(response.status).toBe(200)
      expect(response.data.stats.totalArticles).toBeGreaterThanOrEqual(1)
      expect(response.data.stats.publishedArticles).toBeGreaterThanOrEqual(1)
    })
  })

  describe('GET /api/admin/analytics/charts', () => {
    it('should return chart data with default 30 days', async () => {
      const response = await apiRequest('/api/admin/analytics/charts')

      expect(response.status).toBe(200)
      expect(response.data).toHaveProperty('userRegistrations')
      expect(response.data).toHaveProperty('articleCreations')
      expect(Array.isArray(response.data.userRegistrations)).toBe(true)
      expect(Array.isArray(response.data.articleCreations)).toBe(true)
    })

    it('should return chart data for specified days', async () => {
      const response = await apiRequest('/api/admin/analytics/charts?days=7')

      expect(response.status).toBe(200)
      expect(response.data.userRegistrations.length).toBeLessThanOrEqual(7)
      expect(response.data.articleCreations.length).toBeLessThanOrEqual(7)
    })

    it('should return category distribution', async () => {
      await createTestCategory({ name: 'Dist Cat 1', slug: 'dist-cat-1' })
      await createTestCategory({ name: 'Dist Cat 2', slug: 'dist-cat-2' })

      const response = await apiRequest('/api/admin/analytics/charts')

      expect(response.status).toBe(200)
      expect(response.data).toHaveProperty('categoryDistribution')
      expect(Array.isArray(response.data.categoryDistribution)).toBe(true)
    })

    it('should return difficulty distribution', async () => {
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })

      await createTestArticle(admin!.id, { difficulty: 'beginner', slug: 'beginner-article' })
      await createTestArticle(admin!.id, { difficulty: 'advanced', slug: 'advanced-article' })

      const response = await apiRequest('/api/admin/analytics/charts')

      expect(response.status).toBe(200)
      expect(response.data).toHaveProperty('difficultyDistribution')
      expect(Array.isArray(response.data.difficultyDistribution)).toBe(true)
    })

    it('should return top articles', async () => {
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })

      await createTestArticle(admin!.id, {
        title: 'Popular Article',
        views: 1000,
        slug: 'popular-article',
      })

      const response = await apiRequest('/api/admin/analytics/charts')

      expect(response.status).toBe(200)
      expect(response.data).toHaveProperty('topArticles')
      expect(Array.isArray(response.data.topArticles)).toBe(true)
    })

    it('should return tag usage', async () => {
      await createTestTag({ name: 'Popular Tag', slug: 'popular-tag' })

      const response = await apiRequest('/api/admin/analytics/charts')

      expect(response.status).toBe(200)
      expect(response.data).toHaveProperty('tagUsage')
      expect(Array.isArray(response.data.tagUsage)).toBe(true)
    })
  })
})