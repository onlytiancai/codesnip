import { describe, it, expect, beforeEach } from 'vitest'
import {
  apiRequest,
  loginAsTestUser,
  createTestUser,
  createTestArticle,
  createTestBookmark,
  cleanupDatabase,
  prisma,
} from '../../setup-e2e'

describe('Bookmarks API', () => {
  beforeEach(async () => {
    await cleanupDatabase()
  })

  describe('GET /api/user/bookmarks', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/bookmarks')

      expect(response.status).toBe(401)
    })

    it('should return empty bookmarks for new user', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/bookmarks')

      expect(response.status).toBe(200)
      expect(response.data.bookmarks).toBeDefined()
      expect(Array.isArray(response.data.bookmarks)).toBe(true)
      expect(response.data.pagination).toBeDefined()
    })

    it('should return bookmarks with article details', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })
      const article = await createTestArticle(user!.id, {
        slug: 'bookmarked-article',
        title: 'Bookmarked Article',
        status: 'published',
      })

      // Create bookmark
      await createTestBookmark(user!.id, article.id)

      const response = await apiRequest('/api/user/bookmarks')

      expect(response.status).toBe(200)
      expect(response.data.bookmarks.length).toBe(1)
      expect(response.data.bookmarks[0].title).toBe('Bookmarked Article')
    })

    it('should support pagination', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/bookmarks?page=1&limit=5')

      expect(response.status).toBe(200)
      expect(response.data.pagination.page).toBe(1)
      expect(response.data.pagination.limit).toBe(5)
    })
  })

  describe('POST /api/user/bookmarks/[articleId]', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/bookmarks/1', {
        method: 'POST',
      })

      expect(response.status).toBe(401)
    })

    it('should add bookmark successfully', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })
      const article = await createTestArticle(user!.id, {
        slug: 'to-bookmark',
      })

      const response = await apiRequest(`/api/user/bookmarks/${article.id}`, {
        method: 'POST',
      })

      expect(response.status).toBe(200)
      expect(response.data.bookmark).toBeDefined()
    })

    it('should return existing bookmark if already bookmarked', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })
      const article = await createTestArticle(user!.id, {
        slug: 'already-bookmarked',
      })

      // Add bookmark first
      await createTestBookmark(user!.id, article.id)

      // Try to add again
      const response = await apiRequest(`/api/user/bookmarks/${article.id}`, {
        method: 'POST',
      })

      expect(response.status).toBe(200)
      expect(response.data.message).toBe('Already bookmarked')
    })

    it('should return 404 for non-existent article', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/bookmarks/99999', {
        method: 'POST',
      })

      expect(response.status).toBe(404)
    })
  })

  describe('DELETE /api/user/bookmarks/[articleId]', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/bookmarks/1', {
        method: 'DELETE',
      })

      expect(response.status).toBe(401)
    })

    it('should remove bookmark successfully', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })
      const article = await createTestArticle(user!.id, {
        slug: 'to-unbookmark',
      })

      // Create bookmark first
      await createTestBookmark(user!.id, article.id)

      // Delete bookmark
      const response = await apiRequest(`/api/user/bookmarks/${article.id}`, {
        method: 'DELETE',
      })

      expect(response.status).toBe(200)
      expect(response.data.success).toBe(true)

      // Verify it's deleted
      const bookmark = await prisma.bookmark.findUnique({
        where: {
          userId_articleId: {
            userId: user!.id,
            articleId: article.id,
          },
        },
      })
      expect(bookmark).toBeNull()
    })

    it('should return success even if bookmark does not exist', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/bookmarks/99999', {
        method: 'DELETE',
      })

      expect(response.status).toBe(200)
      expect(response.data.success).toBe(true)
    })
  })
})