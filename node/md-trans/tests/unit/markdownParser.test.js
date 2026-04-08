import { jest } from '@jest/globals';
import { MarkdownParser, createParser } from '../../src/parser/markdownParser.js';

describe('MarkdownParser', () => {
  let parser;

  beforeEach(() => {
    parser = new MarkdownParser();
  });

  describe('parse', () => {
    test('parses simple heading', () => {
      const ast = parser.parse('# Hello');
      expect(ast.type).toBe('root');
      expect(ast.children).toBeDefined();
      expect(ast.children.length).toBeGreaterThan(0);
      expect(ast.children[0].type).toBe('heading');
      expect(ast.children[0].depth).toBe(1);
    });

    test('parses heading with different depths', () => {
      const ast = parser.parse('## Section');
      expect(ast.children[0].type).toBe('heading');
      expect(ast.children[0].depth).toBe(2);
    });

    test('parses paragraph', () => {
      const ast = parser.parse('This is a paragraph.');
      expect(ast.children[0].type).toBe('paragraph');
    });

    test('parses multiple paragraphs', () => {
      const ast = parser.parse('First paragraph.\n\nSecond paragraph.');
      expect(ast.children.length).toBe(2);
      expect(ast.children[0].type).toBe('paragraph');
      expect(ast.children[1].type).toBe('paragraph');
    });

    test('parses blockquote', () => {
      const ast = parser.parse('> This is a quote');
      expect(ast.children[0].type).toBe('blockquote');
    });

    test('parses unordered list', () => {
      const ast = parser.parse('- Item 1\n- Item 2');
      expect(ast.children[0].type).toBe('list');
    });

    test('parses ordered list', () => {
      const ast = parser.parse('1. Item 1\n2. Item 2');
      expect(ast.children[0].type).toBe('list');
    });

    test('parses code block', () => {
      const ast = parser.parse('```javascript\nconst x = 1;\n```');
      expect(ast.children[0].type).toBe('code');
      expect(ast.children[0].lang).toBe('javascript');
    });

    test('parses table', () => {
      const ast = parser.parse('| Header |\n| ------- |\n| Cell |');
      expect(ast.children[0].type).toBe('table');
    });

    test('parses thematic break', () => {
      const ast = parser.parse('---');
      expect(ast.children[0].type).toBe('thematicBreak');
    });

    test('parses emphasis', () => {
      const ast = parser.parse('*italic*');
      expect(ast.children[0].type).toBe('paragraph');
      expect(ast.children[0].children[0].type).toBe('emphasis');
    });

    test('parses strong', () => {
      const ast = parser.parse('**bold**');
      expect(ast.children[0].type).toBe('paragraph');
      expect(ast.children[0].children[0].type).toBe('strong');
    });

    test('parses inline code', () => {
      const ast = parser.parse('Use `code` inline');
      expect(ast.children[0].type).toBe('paragraph');
      expect(ast.children[0].children.some(c => c.type === 'inlineCode')).toBe(true);
    });
  });

  describe('render', () => {
    test('renders heading back to markdown', () => {
      const ast = parser.parse('# Hello');
      const output = parser.render(ast);
      expect(output).toContain('# Hello');
    });

    test('renders paragraph back to markdown', () => {
      const ast = parser.parse('Hello world');
      const output = parser.render(ast);
      expect(output).toContain('Hello world');
    });

    test('renders list back to markdown', () => {
      const ast = parser.parse('- Item 1\n- Item 2');
      const output = parser.render(ast);
      expect(output).toMatch(/[-*] Item 1/);
      expect(output).toMatch(/[-*] Item 2/);
    });
  });

  describe('parseAndRender', () => {
    test('parses and renders in one step', () => {
      const output = parser.parseAndRender('# Test');
      expect(output).toContain('# Test');
    });
  });

  describe('createParser', () => {
    test('creates new parser instance', () => {
      const instance = createParser();
      expect(instance).toBeInstanceOf(MarkdownParser);
    });
  });
});
