import { prisma } from '../../utils/db'

export default defineEventHandler(async (event) => {
  const id = getRouterParam(event, 'id')
  const session = await getUserSession(event)

  if (!session?.user) {
    throw createError({
      statusCode: 401,
      message: 'Not authenticated'
    })
  }

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Job ID is required'
    })
  }

  const job = await prisma.job.findUnique({
    where: { id }
  })

  if (!job) {
    throw createError({
      statusCode: 404,
      message: 'Job not found'
    })
  }

  if (job.userId !== session.user.id) {
    throw createError({
      statusCode: 403,
      message: 'Access denied'
    })
  }

  return { job }
})
