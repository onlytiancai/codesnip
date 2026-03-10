
export default defineEventHandler(async (event) => {
  const id = parseInt(getRouterParam(event, 'id') || '0')

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Invalid article ID'
    })
  }

  const article = await prisma.article.findUnique({
    where: { id },
    include: {
      Category: true,
      ArticleTag: {
        include: {
          Tag: true
        }
      },
      Sentence: {
        orderBy: { order: 'asc' }
      },
      User: {
        select: {
          id: true,
          name: true,
          email: true
        }
      }
    }
  })

  if (!article) {
    throw createError({
      statusCode: 404,
      message: 'Article not found'
    })
  }

  return {
    ...article,
    tags: article.ArticleTag.map(t => t.Tag),
    sentences: article.Sentence
  }
})