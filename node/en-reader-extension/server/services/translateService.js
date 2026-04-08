const axios = require('axios');
const prompts = require('../config/prompts');

class TranslateService {
  constructor() {
    this.cache = new Map();
    this.baseURL = (process.env.ANTHROPIC_BASE_URL || 'https://api.minimax.io/v1') + '/messages';
    this.apiKey = process.env.ANTHROPIC_API_KEY;
    this.model = process.env.ANTHROPIC_MODEL || 'mini-max-text-01';
  }

  buildUserMessage(html, elements) {
    const elementsStr = elements.map(el =>
      `index: ${el.index}, tag: ${el.tag}, text: ${el.text}`
    ).join('\n');

    return prompts.translate.userTemplate
      .replace('{{html}}', html)
      .replace('{{elements}}', elementsStr);
  }

  async translate(html, elements) {
    const cacheKey = JSON.stringify({ html, elements });
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    const userMessage = this.buildUserMessage(html, elements);

    try {
      const response = await axios.post(
        this.baseURL,
        {
          model: this.model,
          messages: [
            { role: 'system', content: prompts.translate.system },
            { role: 'user', content: userMessage }
          ],
          max_tokens: 16000
        },
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.apiKey}`
          },
          timeout: 60000
        }
      );

      const content = response.data.content?.[0]?.text;
      if (!content) {
        throw new Error('No content in API response');
      }

      const result = JSON.parse(content);
      this.cache.set(cacheKey, result);
      return result;
    } catch (error) {
      if (error.response) {
        throw new Error(`Translate API error: ${error.response.status} - ${JSON.stringify(error.response.data)}`);
      }
      throw error;
    }
  }
}

module.exports = new TranslateService();
