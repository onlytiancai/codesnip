// Shared types and constants for the application

// Article difficulty levels
export type DifficultyLevel = 'beginner' | 'intermediate' | 'advanced'

export const DIFFICULTY_LEVELS: Record<DifficultyLevel, { label: string; color: string }> = {
  beginner: { label: '初级', color: 'green' },
  intermediate: { label: '中级', color: 'yellow' },
  advanced: { label: '高级', color: 'red' },
}

// Article categories
export const ARTICLE_CATEGORIES = [
  { id: 'frontend', name: '前端开发' },
  { id: 'backend', name: '后端开发' },
  { id: 'mobile', name: '移动开发' },
  { id: 'devops', name: 'DevOps' },
  { id: 'ai', name: '人工智能' },
  { id: 'database', name: '数据库' },
  { id: 'tools', name: '开发工具' },
  { id: 'career', name: '职业发展' },
] as const

export type ArticleCategory = (typeof ARTICLE_CATEGORIES)[number]['id']

// API response types
export interface ApiResponse<T> {
  data?: T
  error?: string
  message?: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  pageSize: number
  hasMore: boolean
}

// Article related types
export interface ArticleListItem {
  id: string
  title: string
  slug: string
  summary: string | null
  category: string
  coverImage: string | null
  difficulty: string
  publishedAt: string | null
  createdAt: string
}

export interface ArticleDetail extends ArticleListItem {
  content: string
  sentences: SentenceItem[]
}

export interface SentenceItem {
  id: string
  content: string
  translation: string | null
  audioUrl: string | null
  ipa: string | null
  order: number
}

// User related types
export interface UserProfile {
  id: string
  email: string
  name: string | null
  avatar: string | null
  isVip: boolean
}

// Utility functions
export function formatDate(date: string | null | undefined): string {
  if (!date) return ''
  return new Date(date).toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

export function formatDateTime(date: string | null | undefined): string {
  if (!date) return ''
  return new Date(date).toLocaleString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function getDifficultyLabel(difficulty: string): string {
  const labels: Record<string, string> = {
    beginner: '初级',
    intermediate: '中级',
    advanced: '高级',
  }
  return labels[difficulty] || difficulty
}
