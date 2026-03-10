import { describe, it, expect, beforeEach } from 'vitest'
import {
  apiRequest,
  loginAsAdmin,
  createTestCategory,
  createTestArticle,
  cleanupDatabase,
  prisma,
} from '../../setup-e2e'

describe('Admin Categories API', () => {
  beforeEach(async () => {
    await cleanupDatabase()
    await loginAsAdmin()
  })

  describe('GET /api/admin/categories', () => {
    it('should list all categories', async () => {
      await createTestCategory({ name: 'Cat 1', slug: 'cat-1' })
      await createTestCategory({ name: 'Cat 2', slug: 'cat-2' })

      const response = await apiRequest('/api/admin/categories')

      expect(response.status).toBe(200)
      expect(Array.isArray(response.data)).toBe(true)
      expect(response.data.length).toBeGreaterThanOrEqual(2)
    })

    it('should filter by status=active', async () => {
      await createTestCategory({ name: 'Active', slug: 'active', status: 'active' })

      const response = await apiRequest('/api/admin/categories?status=active')

      expect(response.status).toBe(200)
      response.data.forEach((cat: any) => {
        expect(cat.status).toBe('active')
      })
    })
  })

  describe('POST /api/admin/categories', () => {
    it('should create category with valid data', async () => {
      const response = await apiRequest('/api/admin/categories', {
        method: 'POST',
        body: {
          name: 'New Category',
          slug: 'new-category',
          description: 'Test description',
          color: '#ff0000',
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.name).toBe('New Category')
      expect(response.data.slug).toBe('new-category')
      expect(response.data.color).toBe('#ff0000')
    })

    it('should return 400 for duplicate slug', async () => {
      await createTestCategory({ slug: 'duplicate-slug' })

      const response = await apiRequest('/api/admin/categories', {
        method: 'POST',
        body: {
          name: 'Another Category',
          slug: 'duplicate-slug',
        },
      })

      expect(response.status).toBe(400)
    })

    it('should return 400 for duplicate name', async () => {
      await createTestCategory({ name: 'Duplicate Name', slug: 'dup-name' })

      const response = await apiRequest('/api/admin/categories', {
        method: 'POST',
        body: {
          name: 'Duplicate Name',
          slug: 'another-slug',
        },
      })

      expect(response.status).toBe(400)
    })

    it('should use default values', async () => {
      const response = await apiRequest('/api/admin/categories', {
        method: 'POST',
        body: {
          name: 'Default Category',
          slug: 'default-category',
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.color).toBe('#3b82f6')
      expect(response.data.status).toBe('active')
    })
  })

  describe('PUT /api/admin/categories/[id]', () => {
    it('should update category', async () => {
      const category = await createTestCategory({ name: 'Original', slug: 'original' })

      const response = await apiRequest(`/api/admin/categories/${category.id}`, {
        method: 'PUT',
        body: {
          name: 'Updated',
          color: '#00ff00',
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.name).toBe('Updated')
      expect(response.data.color).toBe('#00ff00')
    })

    it('should return 404 for non-existent category', async () => {
      const response = await apiRequest('/api/admin/categories/99999', {
        method: 'PUT',
        body: { name: 'Updated' },
      })

      expect(response.status).toBe(404)
      expect(response.data.message).toBe('Category not found')
    })

    it('should return 400 for duplicate slug', async () => {
      await createTestCategory({ slug: 'existing-slug' })
      const category = await createTestCategory({ slug: 'update-slug-test' })

      const response = await apiRequest(`/api/admin/categories/${category.id}`, {
        method: 'PUT',
        body: { slug: 'existing-slug' },
      })

      expect(response.status).toBe(400)
    })
  })

  describe('DELETE /api/admin/categories/[id]', () => {
    it('should delete empty category', async () => {
      const category = await createTestCategory({ slug: 'delete-test' })

      const response = await apiRequest(`/api/admin/categories/${category.id}`, {
        method: 'DELETE',
      })

      expect(response.status).toBe(200)
      expect(response.data.success).toBe(true)
    })

    it('should return 400 for category with articles', async () => {
      const category = await createTestCategory({ slug: 'with-articles' })
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })
      await createTestArticle(admin!.id, { categoryId: category.id, slug: 'test-article' })

      const response = await apiRequest(`/api/admin/categories/${category.id}`, {
        method: 'DELETE',
      })

      expect(response.status).toBe(400)
      expect(response.data.message).toContain('Cannot delete category')
    })

    it('should return 404 for non-existent category', async () => {
      const response = await apiRequest('/api/admin/categories/99999', {
        method: 'DELETE',
      })

      expect(response.status).toBe(404)
    })
  })
})