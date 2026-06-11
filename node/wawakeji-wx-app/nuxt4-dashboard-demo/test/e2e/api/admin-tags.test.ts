import { describe, it, expect, beforeEach } from 'vitest'
import {
  apiRequest,
  loginAsAdmin,
  createTestTag,
  cleanupDatabase,
  prisma,
} from '../../setup-e2e'

describe('Admin Tags API', () => {
  beforeEach(async () => {
    await cleanupDatabase()
    await loginAsAdmin()
  })

  describe('GET /api/admin/tags', () => {
    it('should list all tags', async () => {
      await createTestTag({ name: 'Tag 1', slug: 'tag-1' })
      await createTestTag({ name: 'Tag 2', slug: 'tag-2' })

      const response = await apiRequest('/api/admin/tags')

      expect(response.status).toBe(200)
      expect(Array.isArray(response.data)).toBe(true)
      expect(response.data.length).toBeGreaterThanOrEqual(2)
    })

    it('should search tags by name', async () => {
      await createTestTag({ name: 'JavaScript', slug: 'javascript' })
      await createTestTag({ name: 'Python', slug: 'python' })

      const response = await apiRequest('/api/admin/tags?search=Java')

      expect(response.status).toBe(200)
      expect(response.data.length).toBeGreaterThan(0)
      response.data.forEach((tag: any) => {
        expect(tag.name.toLowerCase()).toContain('java')
      })
    })
  })

  describe('POST /api/admin/tags', () => {
    it('should create tag with valid data', async () => {
      const response = await apiRequest('/api/admin/tags', {
        method: 'POST',
        body: {
          name: 'New Tag',
          slug: 'new-tag',
          color: '#ff0000',
        },
      })

      expect(response.status).toBe(200)
      expect(response.data.name).toBe('New Tag')
      expect(response.data.slug).toBe('new-tag')
      expect(response.data.color).toBe('#ff0000')
    })

    it('should return 400 for duplicate slug', async () => {
      await createTestTag({ slug: 'duplicate-slug' })

      const response = await apiRequest('/api/admin/tags', {
        method: 'POST',
        body: {
          name: 'Another Tag',
          slug: 'duplicate-slug',
        },
      })

      expect(response.status).toBe(400)
      expect(response.data.message).toContain('slug')
    })

    it('should return 400 for duplicate name', async () => {
      await createTestTag({ name: 'Duplicate Name', slug: 'duplicate-name' })

      const response = await apiRequest('/api/admin/tags', {
        method: 'POST',
        body: {
          name: 'Duplicate Name',
          slug: 'another-slug',
        },
      })

      expect(response.status).toBe(400)
      expect(response.data.message).toContain('name')
    })

    it('should return error for missing name', async () => {
      const response = await apiRequest('/api/admin/tags', {
        method: 'POST',
        body: {
          slug: 'test-slug',
        },
      })

      expect(response.status).toBe(500)
    })
  })

  describe('PUT /api/admin/tags/[id]', () => {
    it('should update tag', async () => {
      const tag = await createTestTag({ name: 'Original', slug: 'original' })

      const response = await apiRequest(`/api/admin/tags/${tag.id}`, {
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

    it('should return 404 for non-existent tag', async () => {
      const response = await apiRequest('/api/admin/tags/99999', {
        method: 'PUT',
        body: { name: 'Updated' },
      })

      expect(response.status).toBe(404)
      expect(response.data.message).toBe('Tag not found')
    })

    it('should return 400 for duplicate slug', async () => {
      await createTestTag({ slug: 'existing-slug' })
      const tag = await createTestTag({ slug: 'update-slug-test' })

      const response = await apiRequest(`/api/admin/tags/${tag.id}`, {
        method: 'PUT',
        body: { slug: 'existing-slug' },
      })

      expect(response.status).toBe(400)
    })

    it('should return 400 for duplicate name', async () => {
      await createTestTag({ name: 'Existing Name', slug: 'existing-name' })
      const tag = await createTestTag({ name: 'Update Name', slug: 'update-name' })

      const response = await apiRequest(`/api/admin/tags/${tag.id}`, {
        method: 'PUT',
        body: { name: 'Existing Name' },
      })

      expect(response.status).toBe(400)
    })
  })

  describe('DELETE /api/admin/tags/[id]', () => {
    it('should delete tag', async () => {
      const tag = await createTestTag({ slug: 'delete-test' })

      const response = await apiRequest(`/api/admin/tags/${tag.id}`, {
        method: 'DELETE',
      })

      expect(response.status).toBe(200)
      expect(response.data.success).toBe(true)
    })

    it('should return 404 for non-existent tag', async () => {
      const response = await apiRequest('/api/admin/tags/99999', {
        method: 'DELETE',
      })

      expect(response.status).toBe(404)
    })

    it('should return 400 for invalid ID', async () => {
      const response = await apiRequest('/api/admin/tags/invalid', {
        method: 'DELETE',
      })

      expect(response.status).toBe(400)
    })
  })
})