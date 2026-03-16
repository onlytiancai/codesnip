import { deleteImportImages } from '../../../../utils/import-queue'

export default defineEventHandler(async (event) => {
  const id = parseInt(getRouterParam(event, 'id') || '0')

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Invalid task ID'
    })
  }

  const task = await prisma.importQueue.findUnique({
    where: { id }
  })

  if (!task) {
    throw createError({
      statusCode: 404,
      message: 'Import task not found'
    })
  }

  // Can only cancel pending or processing tasks
  if (task.status !== 'pending' && task.status !== 'processing') {
    throw createError({
      statusCode: 400,
      message: 'Cannot cancel a completed or failed task'
    })
  }

  // Update status to cancelled
  await prisma.importQueue.update({
    where: { id },
    data: {
      status: 'cancelled',
      updatedAt: new Date()
    }
  })

  // Delete any downloaded images
  const images = await prisma.importImage.findMany({
    where: { queueId: id, localPath: { not: null } }
  })

  await deleteImportImages(images.map(img => img.localPath!))

  // Delete import images records
  await prisma.importImage.deleteMany({
    where: { queueId: id }
  })

  return { success: true }
})