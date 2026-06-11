import { z } from 'zod'

const categorySchema = z.object({
  name: z.string().min(1, 'Name is required'),
  slug: z.string().min(1, 'Slug is required'),
  description: z.string().optional(),
  icon: z.string().optional(),
  color: z.string().default('#3b82f6'),
  status: z.enum(['active', 'inactive']).default('active'),
  sortOrder: z.number().int().default(0)
})

export default defineEventHandler(async (event) => {
  const body = await readBody(event)

  // Validate input
  const data = categorySchema.parse(body)

  // Check if slug already exists
  const existing = await prisma.category.findUnique({
    where: { slug: data.slug }
  })

  if (existing) {
    throw createError({
      statusCode: 400,
      message: 'A category with this slug already exists'
    })
  }

  // Check if name already exists
  const existingName = await prisma.category.findUnique({
    where: { name: data.name }
  })

  if (existingName) {
    throw createError({
      statusCode: 400,
      message: 'A category with this name already exists'
    })
  }

  // Create category
  const category = await prisma.category.create({
    data: {
      name: data.name,
      slug: data.slug,
      description: data.description,
      icon: data.icon,
      color: data.color,
      status: data.status,
      sortOrder: data.sortOrder
    }
  })

  return category
})