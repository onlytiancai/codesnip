import { describe, it, expect, beforeEach } from 'vitest'
import {
  apiRequest,
  loginAsTestUser,
  createTestUser,
  createTestArticle,
  createTestVocabulary,
  cleanupDatabase,
  prisma,
} from '../../setup-e2e'

describe('Vocabulary API', () => {
  beforeEach(async () => {
    await cleanupDatabase()
  })

  describe('GET /api/user/vocabulary', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/vocabulary')

      expect(response.status).toBe(401)
    })

    it('should return empty vocabulary for new user', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/vocabulary')

      expect(response.status).toBe(200)
      expect(response.data.vocabulary).toBeDefined()
      expect(Array.isArray(response.data.vocabulary)).toBe(true)
      expect(response.data.stats).toBeDefined()
    })

    it('should return vocabulary with stats', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })

      // Create vocabulary words
      await createTestVocabulary(user!.id, { word: 'test', progress: 100 })
      await createTestVocabulary(user!.id, { word: 'example', progress: 50 })

      const response = await apiRequest('/api/user/vocabulary')

      expect(response.status).toBe(200)
      expect(response.data.vocabulary.length).toBe(2)
      expect(response.data.stats.totalWords).toBe(2)
      expect(response.data.stats.mastered).toBe(1)
      expect(response.data.stats.learning).toBe(1)
    })

    it('should filter by mastered status', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })

      await createTestVocabulary(user!.id, { word: 'mastered', progress: 100 })
      await createTestVocabulary(user!.id, { word: 'learning', progress: 50 })

      const response = await apiRequest('/api/user/vocabulary?filter=mastered')

      expect(response.status).toBe(200)
      expect(response.data.vocabulary.length).toBe(1)
      expect(response.data.vocabulary[0].progress).toBe(100)
    })

    it('should support sorting', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })

      await createTestVocabulary(user!.id, { word: 'alpha' })
      await createTestVocabulary(user!.id, { word: 'beta' })

      const response = await apiRequest('/api/user/vocabulary?sort=alpha')

      expect(response.status).toBe(200)
      expect(response.data.vocabulary[0].word).toBe('alpha')
    })
  })

  describe('POST /api/user/vocabulary', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/vocabulary', {
        method: 'POST',
        body: { word: 'test', definition: 'A test word' },
      })

      expect(response.status).toBe(401)
    })

    it('should add new vocabulary word', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/vocabulary', {
        method: 'POST',
        body: {
          word: 'serendipity',
          phonetic: '/ˌserənˈdipədē/',
          definition: 'The occurrence of events by chance in a happy way',
          example: 'Finding that book was pure serendipity.',
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.vocabulary.word).toBe('serendipity')
      expect(response.data.vocabulary.progress).toBe(0)
    })

    it('should return 400 for duplicate word', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })

      await createTestVocabulary(user!.id, { word: 'duplicate' })

      const response = await apiRequest('/api/user/vocabulary', {
        method: 'POST',
        body: {
          word: 'duplicate',
          definition: 'Another definition',
        },
      })

      expect(response.status).toBe(400)
      expect(response.data.message).toContain('already exists')
    })

    it('should return 400 for missing required fields', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/vocabulary', {
        method: 'POST',
        body: { word: 'test' }, // Missing definition
      })

      expect(response.status).toBe(400)
    })
  })

  describe('PUT /api/user/vocabulary/[id]', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/vocabulary/1', {
        method: 'PUT',
        body: { progress: 50 },
      })

      expect(response.status).toBe(401)
    })

    it('should update vocabulary progress', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })
      const vocab = await createTestVocabulary(user!.id, { word: 'update-test', progress: 0 })

      const response = await apiRequest(`/api/user/vocabulary/${vocab.id}`, {
        method: 'PUT',
        body: { progress: 75 },
      })

      expect(response.status).toBe(200)
      expect(response.data.vocabulary.progress).toBe(75)
    })

    it('should return 404 for non-existent vocabulary', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/vocabulary/99999', {
        method: 'PUT',
        body: { progress: 50 },
      })

      expect(response.status).toBe(404)
    })

    it('should not allow updating other user vocabulary', async () => {
      await loginAsTestUser()

      // Create another user with vocabulary
      const otherUser = await createTestUser({ email: 'other-vocab@example.com' })
      const otherVocab = await createTestVocabulary(otherUser.id, { word: 'other-word' })

      const response = await apiRequest(`/api/user/vocabulary/${otherVocab.id}`, {
        method: 'PUT',
        body: { progress: 50 },
      })

      expect(response.status).toBe(404)
    })
  })

  describe('DELETE /api/user/vocabulary/[id]', () => {
    it('should return 401 when not logged in', async () => {
      const response = await apiRequest('/api/user/vocabulary/1', {
        method: 'DELETE',
      })

      expect(response.status).toBe(401)
    })

    it('should delete vocabulary word', async () => {
      await loginAsTestUser()

      const user = await prisma.user.findUnique({
        where: { email: 'user@example.com' },
      })
      const vocab = await createTestVocabulary(user!.id, { word: 'to-delete' })

      const response = await apiRequest(`/api/user/vocabulary/${vocab.id}`, {
        method: 'DELETE',
      })

      expect(response.status).toBe(200)
      expect(response.data.success).toBe(true)
    })

    it('should return 404 for non-existent vocabulary', async () => {
      await loginAsTestUser()

      const response = await apiRequest('/api/user/vocabulary/99999', {
        method: 'DELETE',
      })

      expect(response.status).toBe(404)
    })
  })
})