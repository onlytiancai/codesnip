import { ProxyAgent } from 'undici'

export interface TranslateResult {
  text: string
  from: string
  to: string
}

// Proxy configuration from environment
const getProxyAgent = () => {
  const proxyUrl = process.env.HTTPS_PROXY || process.env.https_proxy ||
                   process.env.HTTP_PROXY || process.env.http_proxy ||
                   process.env.ALL_PROXY || process.env.all_proxy

  if (proxyUrl) {
    console.log('Using proxy for translation:', proxyUrl.replace(/:\/\/.*@/, '://***@'))
    return new ProxyAgent(proxyUrl)
  }
  return undefined
}

/**
 * Translate text using Google Translate API directly
 * @param text - Text to translate
 * @param targetLang - Target language code (default: 'zh-CN')
 * @returns Translated text
 */
export async function translateText(
  text: string,
  targetLang: string = 'zh-CN'
): Promise<TranslateResult> {
  if (!text || !text.trim()) {
    return { text: '', from: 'en', to: targetLang }
  }

  try {
    // Use Google Translate's API endpoint directly
    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=${targetLang}&dt=t&q=${encodeURIComponent(text)}`

    const proxyAgent = getProxyAgent()

    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
      },
      // @ts-ignore - dispatcher is for undici
      dispatcher: proxyAgent
    })

    if (!response.ok) {
      throw new Error(`Translation API error: ${response.status}`)
    }

    const data = await response.json()

    // Parse the response format: [[["translated text","original text",null,null,10],...],null,"en",null,null,null,null,[]]
    if (Array.isArray(data) && Array.isArray(data[0])) {
      const translatedText = data[0]
        .filter((item: unknown[]): item is string[] => Array.isArray(item) && typeof item[0] === 'string')
        .map((item: string[]) => item[0])
        .join('')

      const sourceLang = data[2] || 'auto'

      return {
        text: translatedText,
        from: sourceLang,
        to: targetLang
      }
    }

    throw new Error('Invalid translation response format')
  } catch (error) {
    console.error('Translation error:', error)
    throw new Error(`Failed to translate: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

/**
 * Translate multiple texts with rate limiting
 * @param texts - Array of texts to translate
 * @param targetLang - Target language code
 * @param delayMs - Delay between translations in ms (default: 300ms)
 * @returns Array of translated texts
 */
export async function translateBatch(
  texts: string[],
  targetLang: string = 'zh-CN',
  delayMs: number = 300
): Promise<string[]> {
  const results: string[] = []

  for (let i = 0; i < texts.length; i++) {
    const text = texts[i]
    if (!text || !text.trim()) {
      results.push('')
      continue
    }

    try {
      const result = await translateText(text, targetLang)
      results.push(result.text)

      // Rate limiting - wait before next request
      if (i < texts.length - 1) {
        await sleep(delayMs)
      }
    } catch (error) {
      console.error(`Failed to translate text ${i}:`, error)
      // On error, push empty string to maintain array order
      results.push('')
    }
  }

  return results
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}