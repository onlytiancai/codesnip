import { z } from 'zod'

const categorySchema = z.object({
  name: z.string().min(1, 'Name is required').optional(),
  slug: z.string().min(1, 'Slug is required').optional(),
  description: z.string().optional().nullable(),
  icon: z.string().optional().nullable(),
  color: z.string().optional(),
  status: z.enum(['active', 'inactive']).optional(),
  sortOrder: z.number().int().optional()
})

export default defineEventHandler(async (event) => {
  const id = parseInt(getRouterParam(event, 'id') || '0')
  const body = await readBody(event)

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Invalid category ID'
    })
  }

  // Validate input
  const data = categorySchema.parse(body)

  // Check if category exists
  const existing = await prisma.category.findUnique({
    where: { id }
  })

  if (!existing) {
    throw createError({
      statusCode: 404,
      message: 'Category not found'
    })
  }

  // Check slug uniqueness if updating slug
  if (data.slug && data.slug !== existing.slug) {
    const slugExists = await prisma.category.findUnique({
      where: { slug: data.slug }
    })
    if (slugExists) {
      throw createError({
        statusCode: 400,
        message: 'A category with this slug already exists'
      })
    }
  }

  // Check name uniqueness if updating name
  if (data.name && data.name !== existing.name) {
    const nameExists = await prisma.category.findUnique({
      where: { name: data.name }
    })
    if (nameExists) {
      throw createError({
        statusCode: 400,
        message: 'A category with this name already exists'
      })
    }
  }

  // Update category
  const category = await prisma.category.update({
    where: { id },
    data: {
      ...data,
      updatedAt: new Date()
    }
  })

  return category
})