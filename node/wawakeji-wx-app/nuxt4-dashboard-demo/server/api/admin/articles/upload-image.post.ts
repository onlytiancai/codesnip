import { createHash } from 'crypto'
import { mkdir, writeFile } from 'fs/promises'

export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const { imageUrl } = body

  if (!imageUrl) {
    throw createError({
      statusCode: 400,
      message: 'Image URL is required'
    })
  }

  try {
    // Validate URL
    const url = new URL(imageUrl)
    if (!['http:', 'https:'].includes(url.protocol)) {
      throw new Error('Invalid protocol')
    }

    // Fetch image
    const response = await fetch(imageUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
      }
    })

    if (!response.ok) {
      throw createError({
        statusCode: response.status,
        message: `Failed to fetch image: ${response.status} ${response.statusText}`
      })
    }

    const contentType = response.headers.get('content-type') || ''
    if (!contentType.startsWith('image/')) {
      throw createError({
        statusCode: 400,
        message: 'URL does not point to an image'
      })
    }

    // Determine extension
    const ext = contentType.split('/')[1]?.split(';')[0] || 'jpg'
    const buffer = await response.arrayBuffer()

    // Generate unique filename
    const hash = createHash('md5')
      .update(imageUrl)
      .update(Buffer.from(buffer).toString('base64').slice(0, 100))
      .digest('hex')
      .slice(0, 12)

    const filename = `${Date.now()}-${hash}.${ext}`
    const uploadDir = 'public/uploads/images'

    // Ensure directory exists
    await mkdir(uploadDir, { recursive: true })

    // Save file
    await writeFile(`${uploadDir}/${filename}`, Buffer.from(buffer))

    return {
      localUrl: `/uploads/images/${filename}`,
      filename
    }
  } catch (error: any) {
    if (error.statusCode) throw error

    console.error('Upload image error:', error)
    throw createError({
      statusCode: 500,
      message: error.message || 'Failed to upload image'
    })
  }
})