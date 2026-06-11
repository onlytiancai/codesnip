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
      message: 'Tag name is required'
    })
  }

  const slug = body.slug || slugify(body.name)

  const existingTag = await prisma.tag.findFirst({
    where: {
      OR: [
        { name: body.name },
        { slug }
      ]
    }
  })

  if (existingTag) {
    throw createError({
      statusCode: 400,
      message: 'Tag already exists'
    })
  }

  const tag = await prisma.tag.create({
    data: {
      name: body.name,
      slug
    }
  })

  return { tag }
})
