import { describe, it, expect, beforeEach } from 'vitest'
import {
  apiRequest,
  loginAsTestUser,
  loginAsAdmin,
  createTestUser,
  createTestArticle,
  createTestCategory,
  createTestReadingHistory,
  cleanupDatabase,
  prisma,
} from '../../setup-e2e'

describe('Reading History API', () => {
  beforeEach(async () => {
    await cleanupDatabase()
  })

  describe('GET /api/user/history', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/history')

      expect(response.status).toBe(401)
    })

    it('should return empty history for new user', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/history')

      expect(response.status).toBe(200)
      expect(response.data.history).toBeDefined()
      expect(Array.isArray(response.data.history)).toBe(true)
      expect(response.data.pagination).toBeDefined()
    })

    it('should return reading history with pagination', async () => {
      await loginAsTestUser()

      // Get user and create article
      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })
      const category = await createTestCategory()
      const article = await createTestArticle(user!.id, {
        slug: 'history-test-article',
        status: 'published',
      })

      // Create reading history
      await createTestReadingHistory(user!.id, article.id, { progress: 50 })

      const response = await apiRequest('/api/user/history')

      expect(response.status).toBe(200)
      expect(response.data.history.length).toBeGreaterThanOrEqual(1)
      expect(response.data.history[0].progress).toBe(50)
    })
  })

  describe('POST /api/user/history/[articleId]', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/history/1', {
        method: 'POST',
        body: { progress: 50 },
      })

      expect(response.status).toBe(401)
    })

    it('should create new reading history', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })
      const article = await createTestArticle(user!.id, {
        slug: 'new-history-article',
      })

      const response = await apiRequest(`/api/user/history/${article.id}`, {
        method: 'POST',
        body: { progress: 30 },
      })

      expect(response.status).toBe(200)
      expect(response.data.history.progress).toBe(30)
    })

    it('should update existing reading history', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })
      const article = await createTestArticle(user!.id, {
        slug: 'update-history-article',
      })

      // Create initial history
      await createTestReadingHistory(user!.id, article.id, { progress: 30 })

      // Update progress
      const response = await apiRequest(`/api/user/history/${article.id}`, {
        method: 'POST',
        body: { progress: 60 },
      })

      expect(response.status).toBe(200)
      expect(response.data.history.progress).toBe(60)
    })

    it('should set completedAt when progress is 100', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })
      const article = await createTestArticle(user!.id, {
        slug: 'complete-article',
      })

      const response = await apiRequest(`/api/user/history/${article.id}`, {
        method: 'POST',
        body: { progress: 100 },
      })

      expect(response.status).toBe(200)
      expect(response.data.history.progress).toBe(100)
      expect(response.data.history.completedAt).toBeDefined()
    })

    it('should return 404 for non-existent article', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/history/99999', {
        method: 'POST',
        body: { progress: 50 },
      })

      expect(response.status).toBe(404)
    })

    it('should return 400 for invalid progress', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/history/1', {
        method: 'POST',
        body: { progress: 150 }, // Invalid: > 100
      })

      expect(response.status).toBe(400)
    })
  })

  describe('GET /api/user/history/stats', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/history/stats')

      expect(response.status).toBe(401)
    })

    it('should return reading stats', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/history/stats')

      expect(response.status).toBe(200)
      expect(response.data.articlesRead).toBeDefined()
      expect(response.data.totalMinutes).toBeDefined()
      expect(response.data.streak).toBeDefined()
    })
  })
})