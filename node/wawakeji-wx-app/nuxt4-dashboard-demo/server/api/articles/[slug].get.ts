export default defineEventHandler(async (event) => {
  const slug = getRouterParam(event, 'slug')

  if (!slug) {
    throw createError({
      statusCode: 400,
      message: 'Article slug is required'
    })
  }

  const article = await prisma.article.findFirst({
    where: {
      slug,
      status: 'published'
    },
    include: {
      Category: true,
      ArticleTag: {
        include: {
          Tag: true
        }
      },
      User: {
        select: {
          id: true,
          name: true
        }
      },
      Sentence: {
        orderBy: {
          order: 'asc'
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

  // Increment view count
  await prisma.article.update({
    where: { id: article.id },
    data: { views: { increment: 1 } }
  })

  // Transform sentences into paragraphs (group by 2-3 sentences)
  const sentences = article.Sentence.map(s => ({
    en: s.en,
    cn: s.cn
  }))

  const paragraphs = []
  for (let i = 0; i < sentences.length; i += 2) {
    paragraphs.push({
      sentences: sentences.slice(i, i + 2)
    })
  }

  return {
    id: article.id,
    title: article.title,
    slug: article.slug,
    excerpt: article.excerpt,
    cover: article.cover,
    content: article.content,
    difficulty: article.difficulty,
    category: article.Category,
    tags: article.ArticleTag.map(t => t.Tag),
    author: article.User,
    views: article.views + 1, // Return incremented view count
    bookmarks: article.bookmarks,
    readTime: Math.ceil(article.content.split(' ').length / 200),
    createdAt: article.createdAt,
    paragraphs,
    sentences: article.Sentence
  }
})