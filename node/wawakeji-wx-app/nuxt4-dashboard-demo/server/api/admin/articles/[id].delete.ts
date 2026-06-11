
import { deleteImportImages } from '../../../utils/import-queue'

export default defineEventHandler(async (event) => {
  const id = parseInt(getRouterParam(event, 'id') || '0')

  if (!id) {
    throw createError({
      statusCode: 400,
      message: 'Invalid article ID'
    })
  }

  // Check if article exists
  const existing = await prisma.article.findUnique({
    where: { id },
    include: {
      ImportQueue: {
        include: {
          ImportImage: true
        }
      }
    }
  })

  if (!existing) {
    throw createError({
      statusCode: 404,
      message: 'Article not found'
    })
  }

  // If article was imported, clean up images
  if (existing.importTaskId && existing.ImportQueue) {
    // Get local paths of downloaded images
    const imagePaths = existing.ImportQueue.ImportImage
      .filter(img => img.localPath)
      .map(img => img.localPath!)

    // Delete image files from filesystem
    if (imagePaths.length > 0) {
      await deleteImportImages(imagePaths)
    }

    // ImportImage records will be cascade deleted when ImportQueue is deleted
  }

  // Delete article (sentences, tags, and ImportQueue will be cascade deleted)
  await prisma.article.delete({
    where: { id }
  })

  return { success: true }
})