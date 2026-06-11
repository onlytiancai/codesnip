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

  // Set SSE headers
  setResponseHeaders(event, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  })

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()
      let lastProgress = -1
      let lastStage = ''
      let lastStatus = ''
      let checkCount = 0
      const maxChecks = 600 // 10 minutes at 1 second intervals

      const sendEvent = (data: any) => {
        const message = `data: ${JSON.stringify(data)}\n\n`
        controller.enqueue(encoder.encode(message))
      }

      // Send initial state
      sendEvent({
        id: task.id,
        status: task.status,
        stage: task.stage,
        progress: task.progress,
        totalImages: task.totalImages,
        processedImages: task.processedImages
      })

      // Poll for updates
      const pollInterval = setInterval(async () => {
        checkCount++

        // Safety limit
        if (checkCount > maxChecks) {
          clearInterval(pollInterval)
          controller.close()
          return
        }

        try {
          const currentTask = await prisma.importQueue.findUnique({
            where: { id }
          })

          if (!currentTask) {
            clearInterval(pollInterval)
            sendEvent({ error: 'Task not found' })
            controller.close()
            return
          }

          // Send update if anything changed
          if (
            currentTask.progress !== lastProgress ||
            currentTask.stage !== lastStage ||
            currentTask.status !== lastStatus
          ) {
            sendEvent({
              id: currentTask.id,
              status: currentTask.status,
              stage: currentTask.stage,
              progress: currentTask.progress,
              totalImages: currentTask.totalImages,
              processedImages: currentTask.processedImages,
              result: currentTask.result,
              error: currentTask.error
            })

            lastProgress = currentTask.progress
            lastStage = currentTask.stage
            lastStatus = currentTask.status
          }

          // Close stream if task is done
          if (['completed', 'failed', 'cancelled'].includes(currentTask.status)) {
            clearInterval(pollInterval)
            controller.close()
          }
        } catch (error) {
          console.error('SSE poll error:', error)
          clearInterval(pollInterval)
          controller.close()
        }
      }, 1000) // Poll every second

      // Handle client disconnect
      event.node.req.on('close', () => {
        clearInterval(pollInterval)
        controller.close()
      })
    }
  })

  return sendStream(event, stream)
})