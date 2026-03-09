import { z } from 'zod'

const articleSchema = z.object({
  title: z.string().min(1, 'Title is required').optional(),
  slug: z.string().min(1, 'Slug is required').optional(),
  excerpt: z.string().optional().nullable(),
  cover: z.string().optional().nullable(),
  content: z.string().optional().nullable(),
  status: z.enum(['draft', 'published']).optional(),
  difficulty: z.enum(['beginner', 'intermediate', 'advanced']).optional(),
  publishAt: z.string().optional().nullable(),
  metaTitle: z.string().optional().nullable(),
  metaDesc: z.string().optional().nullable(),
  categoryId: z.number().int().optional().nullable(),
  tagIds: z.array(z.number().int()).optional(),
  sentences: z.array(z.object({
    id: z.number().int().optional(),
    order: z.number().int(),
    en: z.string(),
    cn: z.string().optional(),
    audio: z.string().optional()
  })).optional()
})

export default defineEventHandler(async (event) => {
  const id = parseInt(getRouterParam(event, 'id') || '0')
  const body = await readBody(event)

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Invalid article ID'
    })
  }

  // Validate input
  const data = articleSchema.parse(body)

  // Check if article exists
  const existing = await prisma.article.findUnique({
    where: { id }
  })

  if (!existing) {
    throw createError({
      statusCode: 404,
      message: 'Article not found'
    })
  }

  // Check slug uniqueness if updating slug
  if (data.slug && data.slug !== existing.slug) {
    const slugExists = await prisma.article.findUnique({
      where: { slug: data.slug }
    })
    if (slugExists) {
      throw createError({
        statusCode: 400,
        message: 'An article with this slug already exists'
      })
    }
  }

  // Update article
  const updateData: any = {
    ...data,
    publishAt: data.publishAt ? new Date(data.publishAt) : null,
    updatedAt: new Date()
  }

  // Remove tagIds and sentences from direct update, handle separately
  delete updateData.tagIds
  delete updateData.sentences

  // Handle tags update
  if (data.tagIds !== undefined) {
    // Delete existing tags
    await prisma.articleTag.deleteMany({
      where: { articleId: id }
    })
    // Create new tags
    if (data.tagIds.length > 0) {
      await prisma.articleTag.createMany({
        data: data.tagIds.map(tagId => ({
          articleId: id,
          tagId
        }))
      })
    }
  }

  // Handle sentences update
  if (data.sentences !== undefined) {
    // Delete existing sentences
    await prisma.sentence.deleteMany({
      where: { articleId: id }
    })
    // Create new sentences
    if (data.sentences.length > 0) {
      await prisma.sentence.createMany({
        data: data.sentences.map(s => ({
          articleId: id,
          order: s.order,
          en: s.en,
          cn: s.cn,
          audio: s.audio
        }))
      })
    }
  }

  const article = await prisma.article.update({
    where: { id },
    data: updateData,
    include: {
      category: true,
      tags: {
        include: {
          tag: true
        }
      },
      sentences: {
        orderBy: { order: 'asc' }
      }
    }
  })

  return {
    ...article,
    tags: article.tags.map(t => t.tag)
  }
})