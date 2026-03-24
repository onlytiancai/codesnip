import { createScrapeWorker } from '../utils/jobQueue'

console.log('Starting scraper worker...')

const worker = createScrapeWorker()

worker.on('completed', (job) => {
  console.log(`Job ${job.id} completed successfully`)
})

worker.on('failed', (job, err) => {
  console.error(`Job ${job?.id} failed:`, err.message)
})

process.on('SIGTERM', async () => {
  console.log('Shutting down worker...')
  await worker.close()
  process.exit(0)
})

console.log('Scraper worker is ready')
