import { createPreanalysisPrompt } from '../config/prompts.js';
import { getClient } from './llmClient.js';

export class PreAnalyzer {
  constructor(options = {}) {
    this.client = options.client || getClient();
    this.logger = options.logger;
  }

  async analyze(text, options = {}) {
    this.logger?.debug('Starting pre-analysis...');

    const prompt = createPreanalysisPrompt(text);

    try {
      const response = await this.client.complete(prompt, {
        temperature: 0.3,
        max_tokens: 2000,
      });

      const result = this._parseResponse(response);
      this.logger?.debug('Pre-analysis completed', result);
      return result;
    } catch (error) {
      this.logger?.error('Pre-analysis failed', error.message);
      throw error;
    }
  }

  _parseResponse(response) {
    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
    } catch (error) {
      this.logger?.warn('Failed to parse pre-analysis response as JSON');
    }

    return {
      glossary: [],
      context: '',
      warnings: [],
    };
  }

  extractTerms(analysis) {
    return analysis.glossary || [];
  }

  getContext(analysis) {
    return analysis.context || '';
  }

  getWarnings(analysis) {
    return analysis.warnings || [];
  }
}

export function createPreAnalyzer(options = {}) {
  return new PreAnalyzer(options);
}

export default PreAnalyzer;
