import { NodeType, classifyNode } from '../parser/nodeClassifier.js';
import { createTranslationPrompt } from '../config/prompts.js';
import { getClient } from './llmClient.js';
import { createRenderer } from '../processor/bilingualRenderer.js';
import { ConsistencyCache } from '../processor/consistencyCache.js';

export class Translator {
  constructor(options = {}) {
    this.client = options.client || getClient();
    this.cache = options.cache || new ConsistencyCache({ logger: options.logger });
    this.logger = options.logger;
    this.template = options.template;
    this.skipPreanalysis = options.skipPreanalysis || false;
    this.renderer = createRenderer({ logger: options.logger });
  }

  async translate(ast, preanalysis = null, options = {}) {
    const results = [];
    const glossary = preanalysis?.glossary || [];

    if (glossary.length > 0) {
      this.cache.loadFromGlossary(glossary);
    }

    const context = preanalysis?.context || '';

    await this._traverseAndTranslate(ast, results, context);

    return results;
  }

  async _traverseAndTranslate(node, results, context) {
    if (!node || typeof node !== 'object') {
      return;
    }

    const nodeType = classifyNode(node);

    if (nodeType === NodeType.SKIP) {
      results.push({
        node,
        translation: null,
        skipped: true,
      });
      return;
    }

    if (node.type === 'code') {
      results.push({
        node,
        translation: null,
        skipped: true,
      });
      return;
    }

    if (node.type === 'inlineCode') {
      results.push({
        node,
        translation: null,
        skipped: true,
      });
      return;
    }

    if (nodeType === NodeType.TRANSLATABLE || nodeType === NodeType.SPECIAL) {
      const text = this._extractText(node);

      if (!text || text.trim() === '') {
        results.push({
          node,
          translation: '',
          skipped: false,
        });
        return;
      }

      const cached = this.cache.getTranslation(text);
      let translation;

      if (cached) {
        translation = cached;
        this.logger?.debug(`Using cached translation for: "${text.substring(0, 50)}..."`);
      } else {
        translation = await this._translateText(text, context);
        this.cache.set(text, translation);
      }

      results.push({
        node,
        translation,
        skipped: false,
      });
    }

    if (node.children && Array.isArray(node.children)) {
      for (const child of node.children) {
        await this._traverseAndTranslate(child, results, context);
      }
    }
  }

  async _translateText(text, context = '') {
    const prompt = createTranslationPrompt(
      text,
      this.cache.getAll().map(g => ({ term: g.term, translation: g.translation })),
      context
    );

    try {
      const translation = await this.client.complete(prompt, {
        temperature: 0.3,
        max_tokens: 2000,
      });
      const trimmed = translation.trim();
      if (!trimmed) {
        this.logger?.warn('Empty translation received, using original text');
        return text;
      }
      return trimmed;
    } catch (error) {
      this.logger?.error('Translation failed', error.message);
      throw error;
    }
  }

  _extractText(node) {
    if (!node) return '';

    if (node.type === 'text') {
      return node.value || '';
    }

    if (node.type === 'inlineCode') {
      return '';
    }

    if (node.children && Array.isArray(node.children)) {
      return node.children.map(child => this._extractText(child)).join('');
    }

    if (node.value) {
      return node.value;
    }

    return '';
  }

  setCache(cache) {
    this.cache = cache;
  }
}

export function createTranslator(options = {}) {
  return new Translator(options);
}

export default Translator;
