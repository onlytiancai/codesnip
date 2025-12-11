// utils/translation.js - Translation utilities using Ollama API

/**
 * Translate a sentence using Ollama API
 * @param {string} sentence - The sentence to translate
 * @param {Object} options - Translation options
 * @param {Function} options.onProgress - Callback for progress updates
 * @param {Function} options.onComplete - Callback for translation completion
 * @param {Function} options.onError - Callback for translation errors
 * @param {Object} [config] - API configuration
 * @param {string} [config.ollamaApiUrl] - Ollama API URL
 * @param {string} [config.modelName] - Model name to use
 * @param {string} [config.translationPrompt] - Translation prompt template
 * @param {AbortController} [abortController] - AbortController for cancellation
 * @returns {Promise<string>} The translated sentence
 */
export async function translateSentence(sentence, options = {}, config = {}, abortController = null) {
  if (!sentence) {
    options.onProgress?.('');
    options.onComplete?.('');
    return '';
  }

  const { onProgress, onComplete, onError } = options;
  
  // Use default values if config not provided
  const apiUrl = config.ollamaApiUrl || 'http://localhost:11434/api/generate';
  const modelName = config.modelName || 'gemma3:4b';
  const promptTemplate = config.translationPrompt || '请将以下英文句子翻译成中文："{sentence}"';
  const prompt = promptTemplate.replace('{sentence}', sentence);

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      body: JSON.stringify({
        model: modelName,
        prompt: prompt,
        stream: true
      }),
      signal: abortController?.signal
    });

    if (!response.body) {
      throw new Error('Response body is null');
    }

    let fullTranslation = '';
    const decoder = new TextDecoder();
    let buffer = '';

    for await (const chunk of response.body) {
      buffer += decoder.decode(chunk);
      
      // Split buffer into lines
      const lines = buffer.split('\n');
      // Keep the last line (could be incomplete)
      buffer = lines.pop();
      
      // Process each complete line
      for (const line of lines) {
        if (!line.trim()) continue;
        
        try {
          const chunkData = JSON.parse(line);
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
    }

    // Process any remaining data in buffer
    if (buffer.trim()) {
      try {
        const chunkData = JSON.parse(buffer);
        if (chunkData.response) {
          fullTranslation += chunkData.response;
          onProgress?.(fullTranslation);
        }
      } catch (e) {
        console.error('Failed to parse remaining buffer:', e);
      }
    }

    onComplete?.(fullTranslation);
    return fullTranslation;
  } catch (error) {
    // Ignore AbortError since it's expected when canceling
    if (error.name === 'AbortError') {
      return '';
    }
    console.error('Translation error:', error);
    onError?.(error.message);
    throw error;
  }
}