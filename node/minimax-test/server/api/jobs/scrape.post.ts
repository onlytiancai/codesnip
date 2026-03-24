import { prisma } from '../../utils/db'
import { addScrapeJob } from '../../utils/jobQueue'

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)

  if (!session?.user) {
    throw createError({
      statusCode: 401,
      message: 'Not authenticated'
    })
  }

  const body = await readBody(event)

  if (!body.url) {
    throw createError({
      statusCode: 400,
      message: 'URL is required'
    })
  }

  let url: string
  try {
    url = new URL(body.url).toString()
  } catch {
    throw createError({
      statusCode: 400,
      message: 'Invalid URL'
    })
  }

  const job = await prisma.job.create({
    data: {
      type: 'scrape',
      status: 'pending',
      progress: 0,
      data: JSON.stringify({ url }),
      userId: session.user.id
    }
  })

  await addScrapeJob(job.id, url, session.user.id)

  return { job }
})
