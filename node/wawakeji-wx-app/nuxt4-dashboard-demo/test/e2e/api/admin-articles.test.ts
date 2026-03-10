import { describe, it, expect, beforeEach } from 'vitest'
import {
  apiRequest,
  loginAsAdmin,
  createTestUser,
  createTestCategory,
  createTestTag,
  createTestArticle,
  cleanupDatabase,
  prisma,
} from '../../setup-e2e'

describe('Admin Articles API', () => {
  let adminCookies: string

  beforeEach(async () => {
    await cleanupDatabase()
    const loginResult = await loginAsAdmin()
    adminCookies = (globalThis as any).testCookies
  })

  describe('GET /api/admin/articles', () => {
    it('should list articles without filters', async () => {
      const response = await apiRequest('/api/admin/articles')

      expect(response.status).toBe(200)
      expect(response.data.articles).toBeDefined()
      expect(response.data.pagination).toBeDefined()
      expect(Array.isArray(response.data.articles)).toBe(true)
    })

    it('should list articles with pagination', async () => {
      // Create multiple articles
      const category = await createTestCategory()
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })

      for (let i = 0; i < 15; i++) {
        await createTestArticle(admin!.id, {
          slug: `article-${i}`,
          title: `Article ${i}`,
        })
      }

      const response = await apiRequest('/api/admin/articles?page=2&limit=5')

      expect(response.status).toBe(200)
      expect(response.data.articles.length).toBe(5)
      expect(response.data.pagination.page).toBe(2)
      expect(response.data.pagination.totalPages).toBe(3)
    })

    it('should filter by status', async () => {
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })

      await createTestArticle(admin!.id, { status: 'draft', slug: 'draft-article' })
      await createTestArticle(admin!.id, { status: 'published', slug: 'published-article' })

      const response = await apiRequest('/api/admin/articles?status=published')

      expect(response.status).toBe(200)
      response.data.articles.forEach((article: any) => {
        expect(article.status).toBe('published')
      })
    })

    it('should filter by categoryId', async () => {
      const category = await createTestCategory({ name: 'Tech', slug: 'tech' })
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })

      await createTestArticle(admin!.id, {
        categoryId: category.id,
        slug: 'tech-article',
      })
      await createTestArticle(admin!.id, {
        categoryId: null,
        slug: 'no-category-article',
      })

      const response = await apiRequest(`/api/admin/articles?categoryId=${category.id}`)

      expect(response.status).toBe(200)
      response.data.articles.forEach((article: any) => {
        expect(article.categoryId).toBe(category.id)
      })
    })

    it('should filter by difficulty', async () => {
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })

      await createTestArticle(admin!.id, {
        difficulty: 'advanced',
        slug: 'advanced-article',
      })
      await createTestArticle(admin!.id, {
        difficulty: 'beginner',
        slug: 'beginner-article',
      })

      const response = await apiRequest('/api/admin/articles?difficulty=advanced')

      expect(response.status).toBe(200)
      response.data.articles.forEach((article: any) => {
        expect(article.difficulty).toBe('advanced')
      })
    })

    it('should search by title', async () => {
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })

      await createTestArticle(admin!.id, {
        title: 'Special Unique Title',
        slug: 'special-article',
      })
      await createTestArticle(admin!.id, {
        title: 'Another Article',
        slug: 'another-article',
      })

      const response = await apiRequest('/api/admin/articles?search=Unique')

      expect(response.status).toBe(200)
      expect(response.data.articles.length).toBeGreaterThan(0)
      response.data.articles.forEach((article: any) => {
        expect(article.title.toLowerCase()).toContain('unique')
      })
    })
  })

  describe('GET /api/admin/articles/[id]', () => {
    it('should get existing article', async () => {
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })
      const article = await createTestArticle(admin!.id, { slug: 'test-article' })

      const response = await apiRequest(`/api/admin/articles/${article.id}`)

      expect(response.status).toBe(200)
      expect(response.data.id).toBe(article.id)
      expect(response.data.title).toBe(article.title)
    })

    it('should return 404 for non-existent article', async () => {
      const response = await apiRequest('/api/admin/articles/99999')

      expect(response.status).toBe(404)
      expect(response.data.message).toBe('Article not found')
    })

    it('should return 400 for invalid ID', async () => {
      const response = await apiRequest('/api/admin/articles/invalid')

      expect(response.status).toBe(400)
    })
  })

  describe('POST /api/admin/articles', () => {
    it('should create article with valid data', async () => {
      const response = await apiRequest('/api/admin/articles', {
        method: 'POST',
        body: {
          title: 'New Article',
          slug: 'new-article',
          excerpt: 'Test excerpt',
          content: 'Test content',
          status: 'draft',
          difficulty: 'beginner',
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.title).toBe('New Article')
      expect(response.data.slug).toBe('new-article')
    })

    it('should return 401 without auth', async () => {
      // Clear cookies
      ;(globalThis as any).testCookies = undefined

      const response = await apiRequest('/api/admin/articles', {
        method: 'POST',
        body: {
          title: 'New Article',
          slug: 'new-article',
        },
      })

      expect(response.status).toBe(401)
    })

    it('should return 400 for duplicate slug', async () => {
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })
      await createTestArticle(admin!.id, { slug: 'duplicate-slug' })

      const response = await apiRequest('/api/admin/articles', {
        method: 'POST',
        body: {
          title: 'Another Article',
          slug: 'duplicate-slug',
        },
      })

      expect(response.status).toBe(400)
    })

    it('should create article with tags', async () => {
      const tag = await createTestTag({ name: 'Test Tag', slug: 'test-tag' })

      const response = await apiRequest('/api/admin/articles', {
        method: 'POST',
        body: {
          title: 'Article with Tags',
          slug: 'article-with-tags',
          tagIds: [tag.id],
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.tags).toBeDefined()
      expect(response.data.tags.length).toBe(1)
    })

    it('should create article with sentences', async () => {
      const response = await apiRequest('/api/admin/articles', {
        method: 'POST',
        body: {
          title: 'Article with Sentences',
          slug: 'article-with-sentences',
          sentences: [
            { order: 0, en: 'Hello world', cn: '你好世界' },
            { order: 1, en: 'Goodbye world', cn: '再见世界' },
          ],
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.sentences).toBeDefined()
      expect(response.data.sentences.length).toBe(2)
    })
  })

  describe('PUT /api/admin/articles/[id]', () => {
    it('should update article title', async () => {
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })
      const article = await createTestArticle(admin!.id, { slug: 'update-test' })

      const response = await apiRequest(`/api/admin/articles/${article.id}`, {
        method: 'PUT',
        body: {
          title: 'Updated Title',
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.title).toBe('Updated Title')
    })

    it('should return 404 for non-existent article', async () => {
      const response = await apiRequest('/api/admin/articles/99999', {
        method: 'PUT',
        body: { title: 'Updated' },
      })

      expect(response.status).toBe(404)
    })

    it('should return 400 for duplicate slug', async () => {
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })
      await createTestArticle(admin!.id, { slug: 'existing-slug' })
      const article = await createTestArticle(admin!.id, { slug: 'update-slug-test' })

      const response = await apiRequest(`/api/admin/articles/${article.id}`, {
        method: 'PUT',
        body: { slug: 'existing-slug' },
      })

      expect(response.status).toBe(400)
    })

    it('should update tags', async () => {
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })
      const article = await createTestArticle(admin!.id, { slug: 'update-tags-test' })
      const tag = await createTestTag({ name: 'New Tag', slug: 'new-tag' })

      const response = await apiRequest(`/api/admin/articles/${article.id}`, {
        method: 'PUT',
        body: { tagIds: [tag.id] },
      })

      expect(response.status).toBe(200)
      expect(response.data.tags.length).toBe(1)
      expect(response.data.tags[0].id).toBe(tag.id)
    })

    it('should update sentences', async () => {
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })
      const article = await createTestArticle(admin!.id, { slug: 'update-sentences-test' })

      const response = await apiRequest(`/api/admin/articles/${article.id}`, {
        method: 'PUT',
        body: {
          sentences: [{ order: 0, en: 'Updated sentence', cn: '更新的句子' }],
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.sentences.length).toBe(1)
    })
  })

  describe('DELETE /api/admin/articles/[id]', () => {
    it('should delete existing article', async () => {
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })
      const article = await createTestArticle(admin!.id, { slug: 'delete-test' })

      const response = await apiRequest(`/api/admin/articles/${article.id}`, {
        method: 'DELETE',
      })

      expect(response.status).toBe(200)
      expect(response.data.success).toBe(true)

      // Verify deleted
      const checkResponse = await apiRequest(`/api/admin/articles/${article.id}`)
      expect(checkResponse.status).toBe(404)
    })

    it('should return 404 for non-existent article', async () => {
      const response = await apiRequest('/api/admin/articles/99999', {
        method: 'DELETE',
      })

      expect(response.status).toBe(404)
    })

    it('should return 400 for invalid ID', async () => {
      const response = await apiRequest('/api/admin/articles/invalid', {
        method: 'DELETE',
      })

      expect(response.status).toBe(400)
    })
  })
})