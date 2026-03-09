import { z } from 'zod'

const articleSchema = z.object({
  title: z.string().min(1, 'Title is required'),
  slug: z.string().min(1, 'Slug is required'),
  excerpt: z.string().optional(),
  cover: z.string().optional(),
  content: z.string().optional(),
  status: z.enum(['draft', 'published']).default('draft'),
  difficulty: z.enum(['beginner', 'intermediate', 'advanced']).default('beginner'),
  publishAt: z.string().optional(),
  metaTitle: z.string().optional(),
  metaDesc: z.string().optional(),
  categoryId: z.number().int().optional().nullable(),
  tagIds: z.array(z.number().int()).optional(),
  sentences: z.array(z.object({
    order: z.number().int(),
    en: z.string(),
    cn: z.string().optional(),
    audio: z.string().optional()
  })).optional()
})

export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const session = await getUserSession(event)

  if (!session?.user?.id) {
    throw createError({
      statusCode: 401,
      message: 'Unauthorized'
    })
  }

  // Validate input
  const data = articleSchema.parse(body)

  // Check if slug already exists
  const existing = await prisma.article.findUnique({
    where: { slug: data.slug }
  })

  if (existing) {
    throw createError({
      statusCode: 400,
      message: 'An article with this slug already exists'
    })
  }

  // Create article with tags and sentences
  const article = await prisma.article.create({
    data: {
      title: data.title,
      slug: data.slug,
      excerpt: data.excerpt,
      cover: data.cover,
      content: data.content,
      status: data.status,
      difficulty: data.difficulty,
      publishAt: data.publishAt ? new Date(data.publishAt) : null,
      metaTitle: data.metaTitle,
      metaDesc: data.metaDesc,
      categoryId: data.categoryId,
      authorId: session.user.id,
      tags: data.tagIds ? {
        create: data.tagIds.map(tagId => ({
          tag: { connect: { id: tagId } }
        }))
      } : undefined,
      sentences: data.sentences ? {
        createMany: {
          data: data.sentences
        }
      } : undefined
    },
    include: {
      category: true,
      tags: {
        include: {
          tag: true
        }
      },
      sentences: true
    }
  })

  return {
    ...article,
    tags: article.tags.map(t => t.tag)
  }
})