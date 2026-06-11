import { jest } from '@jest/globals';
import { ConsistencyCache, createCache } from '../../src/processor/consistencyCache.js';

describe('ConsistencyCache', () => {
  let cache;

  beforeEach(() => {
    cache = new ConsistencyCache({ maxSize: 100 });
  });

  describe('set and get', () => {
    test('stores and retrieves translation', () => {
      cache.set('API', 'API');
      expect(cache.get('API')).toEqual({
        translation: 'API',
        timestamp: expect.any(Number),
      });
    });

    test('normalizes terms to lowercase', () => {
      cache.set('API', 'API');
      cache.set('api', 'API');
      expect(cache.size()).toBe(1);
    });

    test('trims whitespace from terms', () => {
      cache.set('  API  ', 'API');
      expect(cache.has('API')).toBe(true);
    });

    test('stores entry with timestamp', () => {
      const before = Date.now();
      cache.set('test', '测试');
      const after = Date.now();
      const entry = cache.get('test');
      expect(entry.timestamp).toBeGreaterThanOrEqual(before);
      expect(entry.timestamp).toBeLessThanOrEqual(after);
    });
  });

  describe('has', () => {
    test('returns true for existing term', () => {
      cache.set('API', 'API');
      expect(cache.has('API')).toBe(true);
    });

    test('returns false for non-existing term', () => {
      expect(cache.has('non-existent')).toBe(false);
    });
  });

  describe('getTranslation', () => {
    test('returns translation for existing term', () => {
      cache.set('API', 'API');
      expect(cache.getTranslation('API')).toBe('API');
    });

    test('returns null for non-existing term', () => {
      expect(cache.getTranslation('non-existent')).toBeNull();
    });
  });

  describe('loadFromGlossary', () => {
    test('loads terms from glossary array', () => {
      const glossary = [
        { term: 'API', translation: 'API' },
        { term: 'SDK', translation: 'SDK' },
      ];
      cache.loadFromGlossary(glossary);
      expect(cache.size()).toBe(2);
      expect(cache.getTranslation('API')).toBe('API');
      expect(cache.getTranslation('SDK')).toBe('SDK');
    });

    test('handles empty glossary', () => {
      cache.loadFromGlossary([]);
      expect(cache.size()).toBe(0);
    });

    test('handles non-array input', () => {
      cache.loadFromGlossary(null);
      expect(cache.size()).toBe(0);
    });

    test('skips items without term or translation', () => {
      const glossary = [
        { term: 'API', translation: 'API' },
        { term: 'SDK' },
        { term: 'foo' },
      ];
      cache.loadFromGlossary(glossary);
      expect(cache.size()).toBe(1);
    });
  });

  describe('loadFromPreAnalysis', () => {
    test('loads glossary from preanalysis result', () => {
      const analysis = {
        glossary: [
          { term: 'API', translation: 'API' },
          { term: 'SDK', translation: 'SDK' },
        ],
      };
      cache.loadFromPreAnalysis(analysis);
      expect(cache.size()).toBe(2);
    });

    test('handles empty preanalysis', () => {
      cache.loadFromPreAnalysis({});
      expect(cache.size()).toBe(0);
    });

    test('handles null preanalysis', () => {
      cache.loadFromPreAnalysis(null);
      expect(cache.size()).toBe(0);
    });
  });

  describe('clear', () => {
    test('removes all entries', () => {
      cache.set('a', 'A');
      cache.set('b', 'B');
      cache.clear();
      expect(cache.size()).toBe(0);
    });
  });

  describe('size', () => {
    test('returns correct count', () => {
      expect(cache.size()).toBe(0);
      cache.set('a', 'A');
      expect(cache.size()).toBe(1);
      cache.set('b', 'B');
      expect(cache.size()).toBe(2);
    });
  });

  describe('maxSize enforcement', () => {
    test('evicts oldest entry when max size reached', () => {
      const smallCache = new ConsistencyCache({ maxSize: 2 });
      smallCache.set('a', 'A');
      smallCache.set('b', 'B');
      smallCache.set('c', 'C');
      expect(smallCache.has('a')).toBe(false);
      expect(smallCache.has('b')).toBe(true);
      expect(smallCache.has('c')).toBe(true);
    });
  });

  describe('getAll', () => {
    test('returns all entries as array', () => {
      cache.set('a', 'A');
      cache.set('b', 'B');
      const all = cache.getAll();
      expect(all).toHaveLength(2);
      expect(all[0]).toEqual({
        term: 'a',
        translation: 'A',
        timestamp: expect.any(Number),
      });
    });
  });

  describe('stats', () => {
    test('returns size and maxSize', () => {
      cache.set('a', 'A');
      expect(cache.stats()).toEqual({
        size: 1,
        maxSize: 100,
      });
    });
  });

  describe('createCache', () => {
    test('creates instance with options', () => {
      const logger = { debug: jest.fn() };
      const instance = createCache({ logger, maxSize: 50 });
      expect(instance.stats().maxSize).toBe(50);
    });
  });
});
