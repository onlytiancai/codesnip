import { describe, it, expect, beforeEach } from 'vitest'
import {
  apiRequest,
  loginAsAdmin,
  createTestUser,
  createTestArticle,
  cleanupDatabase,
  prisma,
} from '../../setup-e2e'

describe('Admin Users API', () => {
  beforeEach(async () => {
    await cleanupDatabase()
    await loginAsAdmin()
  })

  describe('GET /api/admin/users', () => {
    it('should list users with pagination', async () => {
      const response = await apiRequest('/api/admin/users')

      expect(response.status).toBe(200)
      expect(response.data.users).toBeDefined()
      expect(response.data.pagination).toBeDefined()
      expect(Array.isArray(response.data.users)).toBe(true)
    })

    it('should filter by role', async () => {
      await createTestUser({ role: 'ADMIN', email: 'admin2@example.com' })
      await createTestUser({ role: 'USER', email: 'user2@example.com' })

      const response = await apiRequest('/api/admin/users?role=ADMIN')

      expect(response.status).toBe(200)
      response.data.users.forEach((user: any) => {
        expect(user.role).toBe('ADMIN')
      })
    })

    it('should search by name/email', async () => {
      await createTestUser({ name: 'John Doe', email: 'john@example.com' })
      await createTestUser({ name: 'Jane Smith', email: 'jane@example.com' })

      const response = await apiRequest('/api/admin/users?search=john')

      expect(response.status).toBe(200)
      expect(response.data.users.length).toBeGreaterThan(0)
    })
  })

  describe('GET /api/admin/users/[id]', () => {
    it('should get user by ID', async () => {
      const user = await createTestUser({ email: 'get-user@example.com' })

      const response = await apiRequest(`/api/admin/users/${user.id}`)

      expect(response.status).toBe(200)
      expect(response.data.id).toBe(user.id)
      expect(response.data.email).toBe(user.email)
    })

    it('should return 404 for non-existent user', async () => {
      const response = await apiRequest('/api/admin/users/99999')

      expect(response.status).toBe(404)
      expect(response.data.message).toBe('User not found')
    })

    it('should return 400 for invalid ID', async () => {
      const response = await apiRequest('/api/admin/users/invalid')

      expect(response.status).toBe(400)
    })
  })

  describe('PUT /api/admin/users/[id]', () => {
    it('should update user role to ADMIN', async () => {
      const user = await createTestUser({ role: 'USER', email: 'update-role@example.com' })

      const response = await apiRequest(`/api/admin/users/${user.id}`, {
        method: 'PUT',
        body: { role: 'ADMIN' },
      })

      expect(response.status).toBe(200)
      expect(response.data.role).toBe('ADMIN')
    })

    it('should update user name', async () => {
      const user = await createTestUser({ name: 'Original Name', email: 'update-name@example.com' })

      const response = await apiRequest(`/api/admin/users/${user.id}`, {
        method: 'PUT',
        body: { name: 'Updated Name' },
      })

      expect(response.status).toBe(200)
      expect(response.data.name).toBe('Updated Name')
    })

    it('should return 404 for non-existent user', async () => {
      const response = await apiRequest('/api/admin/users/99999', {
        method: 'PUT',
        body: { name: 'Updated' },
      })

      expect(response.status).toBe(404)
    })

    it('should return error for invalid role', async () => {
      const user = await createTestUser({ email: 'invalid-role@example.com' })

      const response = await apiRequest(`/api/admin/users/${user.id}`, {
        method: 'PUT',
        body: { role: 'INVALID_ROLE' },
      })

      expect(response.status).toBe(500)
    })
  })

  describe('DELETE /api/admin/users/[id]', () => {
    it('should delete other user', async () => {
      const user = await createTestUser({ email: 'delete-user@example.com' })

      const response = await apiRequest(`/api/admin/users/${user.id}`, {
        method: 'DELETE',
      })

      expect(response.status).toBe(200)
      expect(response.data.success).toBe(true)
    })

    it('should return 400 when deleting self', async () => {
      // Get admin user ID
      const admin = await prisma.user.findUnique({
        where: { email: 'admin@example.com' },
      })

      const response = await apiRequest(`/api/admin/users/${admin!.id}`, {
        method: 'DELETE',
      })

      expect(response.status).toBe(400)
      expect(response.data.message).toContain('cannot delete your own account')
    })

    it('should return 404 for non-existent user', async () => {
      const response = await apiRequest('/api/admin/users/99999', {
        method: 'DELETE',
      })

      expect(response.status).toBe(404)
    })
  })
})