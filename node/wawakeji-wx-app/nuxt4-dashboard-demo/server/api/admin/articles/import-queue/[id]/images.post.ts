import { processImagesForJob } from '../../../../../utils/import-queue'

export default defineEventHandler(async (event) => {
  const id = getRouterParam(event, 'id')

  if (!id) {
    throw createError({
      statusCode: 400,
      statusMessage: 'Job ID is required'
    })
  }

  const jobId = parseInt(id, 10)

  if (isNaN(jobId)) {
    throw createError({
      statusCode: 400,
      statusMessage: 'Invalid job ID'
    })
  }

  try {
    // Start image processing in background
    processImagesForJob(jobId).catch((error) => {
      console.error('Image processing failed:', error)
    })

    return {
      success: true,
      message: 'Image download started'
    }
  } catch (error: any) {
    throw createError({
      statusCode: 500,
      statusMessage: error.message || 'Failed to start image download'
    })
  }
})