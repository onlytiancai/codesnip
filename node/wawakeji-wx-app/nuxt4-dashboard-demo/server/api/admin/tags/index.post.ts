import { z } from 'zod'

const tagSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  slug: z.string().min(1, 'Slug is required'),
  description: z.string().optional(),
  color: z.string().default('#3b82f6')
})

export default defineEventHandler(async (event) => {
  const body = await readBody(event)

  // Validate input
  const data = tagSchema.parse(body)

  // Check if slug already exists
  const existing = await prisma.tag.findUnique({
    where: { slug: data.slug }
  })

  if (existing) {
    throw createError({
      statusCode: 400,
      message: 'A tag with this slug already exists'
    })
  }

  // Check if name already exists
  const existingName = await prisma.tag.findUnique({
    where: { name: data.name }
  })

  if (existingName) {
    throw createError({
      statusCode: 400,
      message: 'A tag with this name already exists'
    })
  }

  // Create tag
  const tag = await prisma.tag.create({
    data: {
      name: data.name,
      slug: data.slug,
      description: data.description,
      color: data.color
    }
  })

  return tag
})