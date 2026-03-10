import { beforeAll, afterAll, beforeEach } from 'vitest'
import { $fetch } from 'ofetch'
import { PrismaBetterSQLite3 } from '@prisma/adapter-better-sqlite3'
import { PrismaClient } from '../generated/prisma/client'
import bcrypt from 'bcryptjs'
import 'dotenv/config'

// Database connection for tests
const adapter = new PrismaBetterSQLite3({ url: process.env.DATABASE_URL! })
const prisma = new PrismaClient({ adapter })

// Test server URL
const baseURL = process.env.NUXT_PUBLIC_BASE_URL || 'http://localhost:3001'

// Helper to make API requests
export async function apiRequest(
  path: string,
  options: RequestInit & { body?: any } = {}
) {
  const url = `${baseURL}${path}`

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  }

  // Add cookie header if we have cookies stored
  if ((globalThis as any).testCookies) {
    headers.Cookie = (globalThis as any).testCookies
  }

  const response = await fetch(url, {
    ...options,
    headers,
    body: options.body ? JSON.stringify(options.body) : undefined,
  })

  // Store cookies from response
  const setCookie = response.headers.get('set-cookie')
  if (setCookie) {
    ;(globalThis as any).testCookies = setCookie
  }

  const data = await response.json().catch(() => null)

  return {
    status: response.status,
    data,
    headers: response.headers,
  }
}

// Helper to create test user
export async function createTestUser(overrides: Partial<{
  email: string
  password: string
  name: string
  role: string
}> = {}) {
  const password = overrides.password || 'password123'
  const hashedPassword = await bcrypt.hash(password, 10)

  const user = await prisma.user.create({
    data: {
      email: overrides.email || `test-${Date.now()}@example.com`,
      name: overrides.name || 'Test User',
      password: hashedPassword,
      role: overrides.role || 'USER',
    },
  })

  return { ...user, plainPassword: password }
}

// Helper to create test admin
export async function createTestAdmin(overrides: Partial<{
  email: string
  password: string
  name: string
}> = {}) {
  return createTestUser({ ...overrides, role: 'ADMIN' })
}

// Helper to login and get session
export async function loginAsUser(email: string, password: string) {
  const result = await apiRequest('/api/auth/login', {
    method: 'POST',
    body: { email, password },
  })
  return result
}

// Helper to login as admin
export async function loginAsAdmin() {
  const admin = await prisma.user.findUnique({
    where: { email: 'admin@example.com' },
  })

  if (!admin) {
    throw new Error('Admin user not found. Run db:seed first.')
  }

  return loginAsUser('admin@example.com', 'admin123')
}

// Helper to create test category
export async function createTestCategory(overrides: Partial<{
  name: string
  slug: string
  description: string
  color: string
}> = {}) {
  const slug = overrides.slug || `test-category-${Date.now()}`

  return prisma.category.create({
    data: {
      name: overrides.name || slug,
      slug,
      description: overrides.description || 'Test category',
      color: overrides.color || '#3b82f6',
    },
  })
}

// Helper to create test tag
export async function createTestTag(overrides: Partial<{
  name: string
  slug: string
  description: string
  color: string
}> = {}) {
  const slug = overrides.slug || `test-tag-${Date.now()}`

  return prisma.tag.create({
    data: {
      name: overrides.name || slug,
      slug,
      description: overrides.description || 'Test tag',
      color: overrides.color || '#3b82f6',
    },
  })
}

// Helper to create test article
export async function createTestArticle(authorId: number, overrides: Partial<{
  title: string
  slug: string
  excerpt: string
  content: string
  status: string
  difficulty: string
  categoryId: number
}> = {}) {
  const slug = overrides.slug || `test-article-${Date.now()}`

  return prisma.article.create({
    data: {
      title: overrides.title || 'Test Article',
      slug,
      excerpt: overrides.excerpt || 'Test excerpt',
      content: overrides.content || 'Test content',
      status: overrides.status || 'draft',
      difficulty: overrides.difficulty || 'beginner',
      categoryId: overrides.categoryId || null,
      authorId,
    },
  })
}

// Clean up database helper
export async function cleanupDatabase() {
  await prisma.sentence.deleteMany()
  await prisma.articleTag.deleteMany()
  await prisma.article.deleteMany()
  await prisma.tag.deleteMany()
  await prisma.category.deleteMany()
  await prisma.account.deleteMany()

  // Don't delete seed users, just test users
  await prisma.user.deleteMany({
    where: {
      email: { notIn: ['admin@example.com', 'user@example.com'] },
    },
  })

  // Clear stored cookies
  ;(globalThis as any).testCookies = undefined
}

// Setup and teardown
beforeAll(async () => {
  // Ensure database is seeded
  const admin = await prisma.user.findUnique({
    where: { email: 'admin@example.com' },
  })

  if (!admin) {
    throw new Error('Database not seeded. Run pnpm db:seed first.')
  }
})

afterAll(async () => {
  await prisma.$disconnect()
})

beforeEach(async () => {
  // Clean up test data before each test
  await cleanupDatabase()
})

// Export prisma for use in tests
export { prisma }