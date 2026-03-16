export default defineEventHandler(async (event) => {
  const id = parseInt(getRouterParam(event, 'id') || '0')

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Invalid task ID'
    })
  }

  const task = await prisma.importQueue.findUnique({
    where: { id },
    include: {
      ImportImage: {
        select: {
          id: true,
          originalUrl: true,
          localPath: true,
          status: true,
          error: true
        }
      }
    }
  })

  if (!task) {
    throw createError({
      statusCode: 404,
      message: 'Import task not found'
    })
  }

  return {
    id: task.id,
    url: task.url,
    status: task.status,
    stage: task.stage,
    progress: task.progress,
    totalImages: task.totalImages,
    processedImages: task.processedImages,
    result: task.result,
    error: task.error,
    createdAt: task.createdAt,
    updatedAt: task.updatedAt,
    images: task.ImportImage
  }
})