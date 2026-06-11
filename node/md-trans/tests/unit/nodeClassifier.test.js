import { jest } from '@jest/globals';
import {
  NodeType,
  classifyNode,
  isTranslatable,
  isSkip,
  isSpecial,
} from '../../src/parser/nodeClassifier.js';

describe('nodeClassifier', () => {
  describe('classifyNode', () => {
    test('classifies code nodes as SKIP', () => {
      const node = { type: 'code', value: 'const x = 1;' };
      expect(classifyNode(node)).toBe(NodeType.SKIP);
    });

    test('classifies inlineCode nodes as SKIP', () => {
      const node = { type: 'inlineCode', value: 'x' };
      expect(classifyNode(node)).toBe(NodeType.SKIP);
    });

    test('classifies html nodes as SKIP', () => {
      const node = { type: 'html', value: '<div>test</div>' };
      expect(classifyNode(node)).toBe(NodeType.SKIP);
    });

    test('classifies definition nodes as SKIP', () => {
      const node = { type: 'definition' };
      expect(classifyNode(node)).toBe(NodeType.SKIP);
    });

    test('classifies footnoteDefinition nodes as SKIP', () => {
      const node = { type: 'footnoteDefinition' };
      expect(classifyNode(node)).toBe(NodeType.SKIP);
    });

    test('classifies heading nodes as TRANSLATABLE', () => {
      const node = { type: 'heading', depth: 1 };
      expect(classifyNode(node)).toBe(NodeType.TRANSLATABLE);
    });

    test('classifies paragraph nodes as TRANSLATABLE', () => {
      const node = { type: 'paragraph' };
      expect(classifyNode(node)).toBe(NodeType.TRANSLATABLE);
    });

    test('classifies listItem nodes as TRANSLATABLE', () => {
      const node = { type: 'listItem' };
      expect(classifyNode(node)).toBe(NodeType.TRANSLATABLE);
    });

    test('classifies tableCell nodes as TRANSLATABLE', () => {
      const node = { type: 'tableCell' };
      expect(classifyNode(node)).toBe(NodeType.TRANSLATABLE);
    });

    test('classifies blockquote nodes as TRANSLATABLE', () => {
      const node = { type: 'blockquote' };
      expect(classifyNode(node)).toBe(NodeType.TRANSLATABLE);
    });

    test('classifies text nodes as TRANSLATABLE', () => {
      const node = { type: 'text', value: 'Hello' };
      expect(classifyNode(node)).toBe(NodeType.TRANSLATABLE);
    });

    test('classifies emphasis nodes as TRANSLATABLE', () => {
      const node = { type: 'emphasis' };
      expect(classifyNode(node)).toBe(NodeType.TRANSLATABLE);
    });

    test('classifies strong nodes as TRANSLATABLE', () => {
      const node = { type: 'strong' };
      expect(classifyNode(node)).toBe(NodeType.TRANSLATABLE);
    });

    test('classifies table nodes as TRANSLATABLE', () => {
      const node = { type: 'table' };
      expect(classifyNode(node)).toBe(NodeType.TRANSLATABLE);
    });

    test('classifies list nodes as TRANSLATABLE', () => {
      const node = { type: 'list' };
      expect(classifyNode(node)).toBe(NodeType.TRANSLATABLE);
    });

    test('classifies link nodes as SPECIAL', () => {
      const node = { type: 'link' };
      expect(classifyNode(node)).toBe(NodeType.SPECIAL);
    });

    test('classifies image nodes as SPECIAL', () => {
      const node = { type: 'image' };
      expect(classifyNode(node)).toBe(NodeType.SPECIAL);
    });

    test('returns SKIP for null node', () => {
      expect(classifyNode(null)).toBe(NodeType.SKIP);
    });

    test('returns SKIP for undefined node', () => {
      expect(classifyNode(undefined)).toBe(NodeType.SKIP);
    });

    test('returns SKIP for node without type', () => {
      expect(classifyNode({})).toBe(NodeType.SKIP);
    });
  });

  describe('isTranslatable', () => {
    test('returns true for translatable nodes', () => {
      expect(isTranslatable({ type: 'paragraph' })).toBe(true);
      expect(isTranslatable({ type: 'heading', depth: 1 })).toBe(true);
      expect(isTranslatable({ type: 'text', value: 'Hello' })).toBe(true);
    });

    test('returns false for skip nodes', () => {
      expect(isTranslatable({ type: 'code' })).toBe(false);
      expect(isTranslatable({ type: 'inlineCode' })).toBe(false);
      expect(isTranslatable({ type: 'html' })).toBe(false);
    });

    test('returns false for special nodes', () => {
      expect(isTranslatable({ type: 'link' })).toBe(false);
      expect(isTranslatable({ type: 'image' })).toBe(false);
    });
  });

  describe('isSkip', () => {
    test('returns true for skip nodes', () => {
      expect(isSkip({ type: 'code' })).toBe(true);
      expect(isSkip({ type: 'inlineCode' })).toBe(true);
      expect(isSkip({ type: 'html' })).toBe(true);
      expect(isSkip({ type: 'definition' })).toBe(true);
    });

    test('returns false for translatable nodes', () => {
      expect(isSkip({ type: 'paragraph' })).toBe(false);
      expect(isSkip({ type: 'heading', depth: 1 })).toBe(false);
    });

    test('returns false for special nodes', () => {
      expect(isSkip({ type: 'link' })).toBe(false);
      expect(isSkip({ type: 'image' })).toBe(false);
    });
  });

  describe('isSpecial', () => {
    test('returns true for special nodes', () => {
      expect(isSpecial({ type: 'link' })).toBe(true);
      expect(isSpecial({ type: 'image' })).toBe(true);
    });

    test('returns false for other nodes', () => {
      expect(isSpecial({ type: 'paragraph' })).toBe(false);
      expect(isSpecial({ type: 'code' })).toBe(false);
    });
  });
});
