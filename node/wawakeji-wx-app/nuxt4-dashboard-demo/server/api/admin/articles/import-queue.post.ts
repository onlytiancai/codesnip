import { processImportJob } from '../../../utils/import-queue'

export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const { url } = body

  // Validate URL
  if (!url) {
    throw createError({
      statusCode: 400,
      message: 'URL is required'
    })
  }

  let targetUrl: URL
  try {
    targetUrl = new URL(url)
    if (!['http:', 'https:'].includes(targetUrl.protocol)) {
      throw new Error('Invalid protocol')
    }
  } catch {
    throw createError({
      statusCode: 400,
      message: 'Invalid URL format'
    })
  }

  // Create import queue record
  const importTask = await prisma.importQueue.create({
    data: {
      url: url
    }
  })

  // Start processing in background (don't await)
  processImportJob(importTask.id).catch(err => {
    console.error(`Background import job ${importTask.id} failed:`, err)
  })

  return {
    taskId: importTask.id,
    status: importTask.status,
    stage: importTask.stage
  }
})