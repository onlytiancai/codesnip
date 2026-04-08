export class ConsistencyCache {
  constructor(options = {}) {
    this.cache = new Map();
    this.maxSize = options.maxSize || 10000;
    this.logger = options.logger;
  }

  set(term, translation) {
    if (!translation || translation.trim() === '') {
      return;
    }
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    const normalizedTerm = term.toLowerCase().trim();
    this.cache.set(normalizedTerm, {
      translation,
      timestamp: Date.now(),
    });
  }

  get(term) {
    const normalizedTerm = term.toLowerCase().trim();
    return this.cache.get(normalizedTerm);
  }

  has(term) {
    const normalizedTerm = term.toLowerCase().trim();
    return this.cache.has(normalizedTerm);
  }

  getTranslation(term) {
    const entry = this.get(term);
    return entry ? entry.translation : null;
  }

  loadFromGlossary(glossary) {
    if (!Array.isArray(glossary)) {
      return;
    }

    for (const item of glossary) {
      if (item.term && item.translation) {
        this.set(item.term, item.translation);
      }
    }

    this.logger?.debug(`Loaded ${glossary.length} terms from glossary`);
  }

  loadFromPreAnalysis(analysis) {
    if (analysis && analysis.glossary) {
      this.loadFromGlossary(analysis.glossary);
    }
  }

  clear() {
    this.cache.clear();
  }

  size() {
    return this.cache.size;
  }

  getAll() {
    return Array.from(this.cache.entries()).map(([term, entry]) => ({
      term,
      translation: entry.translation,
      timestamp: entry.timestamp,
    }));
  }

  stats() {
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
    };
  }
}

let cacheInstance = null;

export function createCache(options = {}) {
  cacheInstance = new ConsistencyCache(options);
  return cacheInstance;
}

export function getCache() {
  if (!cacheInstance) {
    cacheInstance = new ConsistencyCache();
  }
  return cacheInstance;
}

export default ConsistencyCache;
