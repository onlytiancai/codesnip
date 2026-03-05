import { H3Event } from 'h3'
import { prisma } from '~/server/db'

export default defineEventHandler(async (event) => {
  try {
    const slug = event.context.params?.slug

    if (!slug) {
      throw createError({
        statusCode: 400,
        message: 'Article slug is required',
      })
    }

    const article = await prisma.article.findFirst({
      where: {
        slug,
        published: true,
      },
      include: {
        sentences: {
          orderBy: { order: 'asc' },
          select: {
            id: true,
            content: true,
            translation: true,
            audioUrl: true,
            ipa: true,
            order: true,
          },
        },
      },
    })

    if (!article) {
      throw createError({
        statusCode: 404,
        message: 'Article not found',
      })
    }

    return {
      id: article.id,
      title: article.title,
      slug: article.slug,
      summary: article.summary,
      content: article.content,
      category: article.category,
      coverImage: article.coverImage,
      difficulty: article.difficulty,
      publishedAt: article.publishedAt,
      createdAt: article.createdAt,
      sentences: article.sentences,
    }
  } catch (error) {
    console.error('Error fetching article:', error)
    if (error.statusCode) {
      throw error
    }
    throw createError({
      statusCode: 500,
      message: 'Failed to fetch article',
    })
  }
})
