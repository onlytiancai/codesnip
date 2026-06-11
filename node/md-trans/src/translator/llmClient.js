import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';

export class LLMClient {
  constructor(options = {}) {
    this.mode = options.mode || 'anthropic'; // 'anthropic' or 'openai'
    this.timeout = options.timeout || 120000;
    this.maxRetries = options.maxRetries || 3;
    this.retryDelay = options.retryDelay || 1000;
    this.logger = options.logger;

    if (this.mode === 'anthropic') {
      this.apiKey = options.apiKey || process.env.ANTHROPIC_API_KEY;
      this.baseURL = options.baseURL || process.env.ANTHROPIC_BASE_URL || 'https://api.minimaxi.com/anthropic';
      this.model = options.model || process.env.ANTHROPIC_MODEL || 'MiniMax-M2.7';
      this.sdkClient = new Anthropic({
        baseURL: this.baseURL,
        apiKey: this.apiKey,
        timeout: this.timeout,
      });
    } else {
      this.apiKey = options.apiKey || process.env.OPENAI_API_KEY;
      this.baseURL = options.baseURL || process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
      this.model = options.model || process.env.OPENAI_MODEL || 'gpt-4';
      this.sdkClient = new OpenAI({
        apiKey: this.apiKey,
        baseURL: this.baseURL,
        timeout: this.timeout,
      });
    }
  }

  async request(messages, options = {}) {
    this.logger?.debug(`[${this.mode.toUpperCase()}] >>> LLM Request`);

    let lastError;
    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const response = this.mode === 'anthropic'
          ? await this._anthropicRequest(messages, options)
          : await this._openaiRequest(messages, options);
        return response;
      } catch (error) {
        lastError = error;
        const status = error.status || error.response?.status;
        if (status === 429) {
          const delay = this.retryDelay * Math.pow(2, attempt);
          this.logger?.debug(`Rate limited (429), retrying in ${delay}ms...`);
          await this._sleep(delay);
          continue;
        }
        if (status === 529) {
          const delay = this.retryDelay * Math.pow(2, attempt + 1);
          this.logger?.debug(`Server overloaded (529), retrying in ${delay}ms... (attempt ${attempt + 1}/${this.maxRetries})`);
          await this._sleep(delay);
          continue;
        }
        if (status >= 500) {
          const delay = this.retryDelay * Math.pow(2, attempt);
          this.logger?.debug(`Server error (${status}), retrying in ${delay}ms...`);
          await this._sleep(delay);
          continue;
        }
        throw error;
      }
    }
    throw lastError;
  }

  async _anthropicRequest(messages, options = {}) {
    const systemPrompt = messages.find(m => m.role === 'system')?.content || '';
    const userMessages = messages.filter(m => m.role !== 'system');

    const body = {
      model: this.model,
      max_tokens: options.max_tokens || 2000,
      system: systemPrompt || undefined,
      messages: userMessages.map(m => ({
        role: m.role,
        content: [{ type: 'text', text: m.content }],
      })),
    };

    this.logger?.debug('>>> Anthropic Request:', JSON.stringify(body, null, 2));

    const response = await this.sdkClient.messages.create(body);

    this.logger?.debug('<<< Anthropic Response:', JSON.stringify(response, null, 2));
    return response;
  }

  async _openaiRequest(messages, options = {}) {
    const body = {
      model: this.model,
      messages,
      temperature: options.temperature ?? 0.3,
    };

    if (options.max_tokens) {
      body.max_tokens = options.max_tokens;
    }

    this.logger?.debug('>>> OpenAI Request:', JSON.stringify(body, null, 2));

    const response = await this.sdkClient.chat.completions.create(body);

    this.logger?.debug('<<< OpenAI Response:', JSON.stringify(response, null, 2));
    return response;
  }

  async translate(text, systemPrompt, options = {}) {
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: text },
    ];

    const response = await this.request(messages, options);
    return this._extractText(response);
  }

  async complete(prompt, options = {}) {
    const messages = [{ role: 'user', content: prompt }];
    const response = await this.request(messages, options);
    return this._extractText(response);
  }

  _extractText(response) {
    if (this.mode === 'anthropic') {
      if (!response?.content) return '';
      const textBlocks = response.content.filter(block => block.type === 'text');
      return textBlocks.map(block => block.text).join('');
    } else {
      return response?.choices?.[0]?.message?.content || '';
    }
  }

  _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

let clientInstance = null;

export function createClient(options = {}) {
  clientInstance = new LLMClient(options);
  return clientInstance;
}

export function getClient() {
  if (!clientInstance) {
    clientInstance = new LLMClient();
  }
  return clientInstance;
}

export default LLMClient;
