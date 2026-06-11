import { jest } from '@jest/globals';
import { BilingualRenderer, createRenderer } from '../../src/processor/bilingualRenderer.js';

describe('BilingualRenderer', () => {
  let renderer;

  beforeEach(() => {
    renderer = new BilingualRenderer();
  });

  describe('_getNodeText', () => {
    test('extracts text from text node', () => {
      const node = { type: 'text', value: 'Hello world' };
      expect(renderer._getNodeText(node)).toBe('Hello world');
    });

    test('extracts combined text from children', () => {
      const node = {
        type: 'paragraph',
        children: [
          { type: 'text', value: 'Hello' },
          { type: 'text', value: ' ' },
          { type: 'text', value: 'world' },
        ],
      };
      expect(renderer._getNodeText(node)).toBe('Hello world');
    });

    test('returns empty string for null node', () => {
      expect(renderer._getNodeText(null)).toBe('');
    });

    test('returns empty string for undefined node', () => {
      expect(renderer._getNodeText(undefined)).toBe('');
    });

    test('handles nested structure', () => {
      const node = {
        type: 'paragraph',
        children: [
          {
            type: 'emphasis',
            children: [{ type: 'text', value: 'italic' }],
          },
          { type: 'text', value: ' normal' },
        ],
      };
      expect(renderer._getNodeText(node)).toBe('italic normal');
    });
  });

  describe('renderNode', () => {
    test('renders text node with translation', () => {
      const node = { type: 'text', value: 'Hello' };
      expect(renderer.renderNode(node, '你好')).toBe('你好');
    });

    test('renders heading with depth 1', () => {
      const node = {
        type: 'heading',
        depth: 1,
        children: [{ type: 'text', value: 'Introduction' }],
      };
      const result = renderer.renderNode(node, '介绍');
      expect(result).toContain('# Introduction');
      expect(result).toContain('# 介绍');
    });

    test('renders heading with depth 2', () => {
      const node = {
        type: 'heading',
        depth: 2,
        children: [{ type: 'text', value: 'Section' }],
      };
      const result = renderer.renderNode(node, '章节');
      expect(result).toContain('## Section');
      expect(result).toContain('## 章节');
    });

    test('renders paragraph', () => {
      const node = {
        type: 'paragraph',
        children: [{ type: 'text', value: 'Some text' }],
      };
      const result = renderer.renderNode(node, '一些文本');
      expect(result).toContain('Some text');
      expect(result).toContain('一些文本');
    });

    test('renders code block', () => {
      const node = {
        type: 'code',
        lang: 'javascript',
        value: 'const x = 1;',
      };
      const result = renderer.renderNode(node, '');
      expect(result).toContain('```javascript');
      expect(result).toContain('const x = 1;');
      expect(result).toContain('```');
    });

    test('renders inline code', () => {
      const node = {
        type: 'inlineCode',
        value: 'x',
      };
      expect(renderer.renderNode(node, '')).toBe('x');
    });

    test('renders thematic break', () => {
      const node = { type: 'thematicBreak' };
      expect(renderer.renderNode(node, '')).toBe('---');
    });
  });

  describe('renderParallel', () => {
    test('renders heading in parallel format', () => {
      const heading = {
        type: 'heading',
        depth: 1,
        children: [{ type: 'text', value: 'Title' }],
      };
      const result = renderer.renderParallel(heading, '标题');
      expect(result).toContain('# Title');
      expect(result).toContain('# 标题');
    });

    test('renders paragraph in parallel format', () => {
      const paragraph = {
        type: 'paragraph',
        children: [{ type: 'text', value: 'Content' }],
      };
      const result = renderer.renderParallel(paragraph, '内容');
      expect(result).toContain('Content');
      expect(result).toContain('内容');
    });

    test('renders blockquote with > prefix', () => {
      const blockquote = {
        type: 'blockquote',
        children: [{ type: 'text', value: 'Quote' }],
      };
      const result = renderer.renderParallel(blockquote, '引用');
      expect(result).toContain('> Quote');
      expect(result).toContain('> 引用');
    });

    test('renders code block', () => {
      const code = {
        type: 'code',
        lang: 'python',
        value: 'print("hello")',
      };
      const result = renderer.renderParallel(code, '');
      expect(result).toContain('```python');
      expect(result).toContain('print("hello")');
    });
  });

  describe('createRenderer', () => {
    test('creates instance with logger option', () => {
      const logger = { debug: jest.fn() };
      const instance = createRenderer({ logger });
      expect(instance).toBeInstanceOf(BilingualRenderer);
    });
  });
});
