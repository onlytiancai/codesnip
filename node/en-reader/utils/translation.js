// utils/translation.js - Translation utilities using Ollama API

/**
 * Translate a sentence using Ollama API
 * @param {string} sentence - The sentence to translate
 * @param {Object} options - Translation options
 * @param {Function} options.onProgress - Callback for progress updates
 * @param {Function} options.onComplete - Callback for translation completion
 * @param {Function} options.onError - Callback for translation errors
 * @returns {Promise<string>} The translated sentence
 */
export async function translateSentence(sentence, options = {}) {
  if (!sentence) {
    options.onProgress?.('');
    options.onComplete?.('');
    return '';
  }

  const { onProgress, onComplete, onError } = options;

  try {
    const response = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      body: JSON.stringify({
        model: "gemma3:4b",
        prompt: `请将以下英文句子翻译成中文："${sentence}"`,
        stream: true
      })
    });

    if (!response.body) {
      throw new Error('Response body is null');
    }

    let fullTranslation = '';
    const decoder = new TextDecoder();

    for await (const chunk of response.body) {
      const chunkText = decoder.decode(chunk);
      try {
        const chunkData = JSON.parse(chunkText);
        if (chunkData.response) {
          fullTranslation += chunkData.response;
          onProgress?.(fullTranslation);
        }
        if (chunkData.done) {
          break;
        }
      } catch (e) {
        console.error('Failed to parse chunk:', e);
      }
    }

    onComplete?.(fullTranslation);
    return fullTranslation;
  } catch (error) {
    console.error('Translation error:', error);
    onError?.(error.message);
    throw error;
  }
}