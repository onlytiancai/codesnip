import { describe, it, expect, beforeEach } from 'vitest'
import {
  apiRequest,
  loginAsTestUser,
  loginAsAdmin,
  createTestUser,
  createTestArticle,
  createTestCategory,
  createTestReadingHistory,
  createTestBookmark,
  createTestVocabulary,
  cleanupDatabase,
  prisma,
} from '../../setup-e2e'

describe('User Profile API', () => {
  beforeEach(async () => {
    await cleanupDatabase()
  })

  describe('GET /api/user/profile', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/profile')

      expect(response.status).toBe(401)
      expect(response.data.message).toBe('Unauthorized')
    })

    it('should return user profile with stats', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/profile')

      expect(response.status).toBe(200)
      expect(response.data.user).toBeDefined()
      expect(response.data.user.email).toBe('user@example.com')
      expect(response.data.stats).toBeDefined()
      expect(response.data.membership).toBeDefined()
    })

    it('should return correct reading stats', async () => {
      await loginAsTestUser()

      // Get user ID
      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })

      // Create category and article
      const category = await createTestCategory()
      const article = await createTestArticle(user!.id, {
        slug: 'test-article-stats',
        status: 'published',
        content: 'This is a test article content with some words. ' + 'word '.repeat(200),
      })

      // Create reading history
      await createTestReadingHistory(user!.id, article.id, { progress: 100 })

      const response = await apiRequest('/api/user/profile')

      expect(response.status).toBe(200)
      expect(response.data.stats.articlesRead).toBeGreaterThanOrEqual(1)
    })
  })

  describe('PUT /api/user/profile', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/profile', {
        method: 'PUT',
        body: { name: 'Updated Name' },
      })

      expect(response.status).toBe(401)
    })

    it('should update user name', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/profile', {
        method: 'PUT',
        body: { name: 'Updated Test User' },
      })

      expect(response.status).toBe(200)
      expect(response.data.user.name).toBe('Updated Test User')
    })

    it('should update user bio', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/profile', {
        method: 'PUT',
        body: { bio: 'This is my bio' },
      })

      expect(response.status).toBe(200)
    })

    it('should return 400 for invalid data', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/profile', {
        method: 'PUT',
        body: { name: '' }, // Empty name should fail
      })

      expect(response.status).toBe(400)
    })
  })
})