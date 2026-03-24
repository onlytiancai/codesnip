import { Queue, Worker, Job as BullJob } from 'bullmq'
import Redis from 'ioredis'
import { scrapeUrl } from './scraper'
import { downloadImage } from './imageDownloader'
import { prisma } from './db'

const connection = new Redis(process.env.REDIS_URL || 'redis://localhost:6379', {
  maxRetriesPerRequest: null
})

export const scrapeQueue = new Queue('scrape', { connection })

export interface ScrapeJobData {
  jobId: string
  url: string
  userId: string
}

export async function addScrapeJob(jobId: string, url: string, userId: string) {
  await scrapeQueue.add('scrape', {
    jobId,
    url,
    userId
  })
}

export function createScrapeWorker() {
  return new Worker<ScrapeJobData>(
    'scrape',
    async (job: BullJob<ScrapeJobData>) => {
      const { jobId, url, userId } = job.data

      try {
        await prisma.job.update({
          where: { id: jobId },
          data: { status: 'active', progress: 10 }
        })

        await job.updateProgress(10)

        const scraped = await scrapeUrl(url)
        await job.updateProgress(30)

        await prisma.job.update({
          where: { id: jobId },
          data: { progress: 30 }
        })

        const article = await prisma.article.create({
          data: {
            title: scraped.title,
            url: url,
            content: scraped.content,
            description: scraped.description,
            userId: userId,
            isPublished: false
          }
        })

        await job.updateProgress(50)
        await prisma.job.update({
          where: { id: jobId },
          data: { progress: 50 }
        })

        const downloadedImages: { url: string; localPath: string }[] = []
        for (let i = 0; i < scraped.images.length; i++) {
          const imgUrl = scraped.images[i]
          const downloaded = await downloadImage(imgUrl, article.id)
          downloadedImages.push(downloaded)

          await prisma.image.create({
            data: {
              url: downloaded.originalUrl,
              localPath: downloaded.localPath,
              articleId: article.id
            }
          })

          const imgProgress = 50 + Math.round(((i + 1) / scraped.images.length) * 20)
          await job.updateProgress(imgProgress)
          await prisma.job.update({
            where: { id: jobId },
            data: { progress: imgProgress }
          })
        }

        let finalContent = scraped.content
        for (const img of downloadedImages) {
          finalContent = finalContent.replace(new RegExp(img.originalUrl.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), img.localPath)
        }

        await prisma.article.update({
          where: { id: article.id },
          data: { content: finalContent }
        })

        await job.updateProgress(90)
        await prisma.job.update({
          where: { id: jobId },
          data: { progress: 90 }
        })

        await prisma.job.update({
          where: { id: jobId },
          data: {
            status: 'completed',
            progress: 100,
            result: JSON.stringify({ articleId: article.id })
          }
        })

        return { articleId: article.id }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error'
        await prisma.job.update({
          where: { id: jobId },
          data: {
            status: 'failed',
            error: errorMessage
          }
        })
        throw error
      }
    },
    { connection }
  )
}
