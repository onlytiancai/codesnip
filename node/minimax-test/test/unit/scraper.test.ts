import { describe, it, expect } from 'vitest'
import { scrapeUrl } from '../../server/utils/scraper'

describe('Scraper', () => {
  it('should handle invalid URLs gracefully', async () => {
    try {
      await scrapeUrl('not-a-valid-url')
      expect.fail('Should have thrown an error')
    } catch (error) {
      expect(error).toBeDefined()
    }
  })

  it('should return expected structure for valid scrape', async () => {
    const result = await scrapeUrl('https://code.claude.com/docs/en/sub-agents')

    expect(result).toHaveProperty('title')
    expect(result).toHaveProperty('content')
    expect(result).toHaveProperty('description')
    expect(result).toHaveProperty('images')

    expect(typeof result.title).toBe('string')
    expect(typeof result.content).toBe('string')
    expect(Array.isArray(result.images)).toBe(true)
  }, 30000)
})
