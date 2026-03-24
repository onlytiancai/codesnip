import { prisma } from '../../utils/db'

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)

  if (!session?.user) {
    throw createError({
      statusCode: 401,
      message: 'Not authenticated'
    })
  }

  const body = await readBody(event)

  if (!body.name) {
    throw createError({
      statusCode: 400,
      message: 'Category name is required'
    })
  }

  const slug = body.slug || slugify(body.name)

  const existingCategory = await prisma.category.findFirst({
    where: {
      OR: [
        { name: body.name },
        { slug }
      ]
    }
  })

  if (existingCategory) {
    throw createError({
      statusCode: 400,
      message: 'Category already exists'
    })
  }

  const category = await prisma.category.create({
    data: {
      name: body.name,
      slug
    }
  })

  return { category }
})
