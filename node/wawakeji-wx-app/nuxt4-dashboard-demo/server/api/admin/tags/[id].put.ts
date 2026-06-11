import { z } from 'zod'

const tagSchema = z.object({
  name: z.string().min(1, 'Name is required').optional(),
  slug: z.string().min(1, 'Slug is required').optional(),
  description: z.string().optional().nullable(),
  color: z.string().optional()
})

export default defineEventHandler(async (event) => {
  const id = parseInt(getRouterParam(event, 'id') || '0')
  const body = await readBody(event)

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Invalid tag ID'
    })
  }

  // Validate input
  const data = tagSchema.parse(body)

  // Check if tag exists
  const existing = await prisma.tag.findUnique({
    where: { id }
  })

  if (!existing) {
    throw createError({
      statusCode: 404,
      message: 'Tag not found'
    })
  }

  // Check slug uniqueness if updating slug
  if (data.slug && data.slug !== existing.slug) {
    const slugExists = await prisma.tag.findUnique({
      where: { slug: data.slug }
    })
    if (slugExists) {
      throw createError({
        statusCode: 400,
        message: 'A tag with this slug already exists'
      })
    }
  }

  // Check name uniqueness if updating name
  if (data.name && data.name !== existing.name) {
    const nameExists = await prisma.tag.findUnique({
      where: { name: data.name }
    })
    if (nameExists) {
      throw createError({
        statusCode: 400,
        message: 'A tag with this name already exists'
      })
    }
  }

  // Update tag
  const tag = await prisma.tag.update({
    where: { id },
    data: {
      ...data,
      updatedAt: new Date()
    }
  })

  return tag
})