import { writeFile, mkdir } from 'fs/promises'
import { join } from 'path'
import { existsSync } from 'fs'

export interface DownloadedImage {
  originalUrl: string
  localPath: string
}

export async function downloadImage(url: string, articleId: string): Promise<DownloadedImage> {
  try {
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
      }
    })

    if (!response.ok) {
      throw new Error(`Failed to download image: ${response.statusText}`)
    }

    const buffer = await response.arrayBuffer()
    const contentType = response.headers.get('content-type') || 'image/jpeg'
    const extension = contentType.split('/')[1] || 'jpg'

    const timestamp = Date.now()
    const filename = `${timestamp}.${extension}`
    const dirPath = join('public', 'images', 'articles', articleId)

    if (!existsSync(dirPath)) {
      await mkdir(dirPath, { recursive: true })
    }

    const localPath = join(dirPath, filename)
    const publicPath = `/images/articles/${articleId}/${filename}`

    await writeFile(localPath, Buffer.from(buffer))

    return {
      originalUrl: url,
      localPath: publicPath
    }
  } catch (error) {
    console.error(`Failed to download image from ${url}:`, error)
    return {
      originalUrl: url,
      localPath: url
    }
  }
}

export async function deleteArticleImages(articleId: string): Promise<void> {
  const { rm } = await import('fs/promises')
  const dirPath = join('public', 'images', 'articles', articleId)

  try {
    if (existsSync(dirPath)) {
      await rm(dirPath, { recursive: true })
    }
  } catch (error) {
    console.error(`Failed to delete images for article ${articleId}:`, error)
  }
}
