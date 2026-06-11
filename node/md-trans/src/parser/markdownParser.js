import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkStringify from 'remark-stringify';

export class MarkdownParser {
  constructor(options = {}) {
    this.processor = unified()
      .use(remarkParse)
      .use(remarkGfm)
      .use(remarkStringify, {
        listItemIndent: 'one',
        tightDefinitions: true,
      });
  }

  parse(markdown) {
    const ast = this.processor.parse(markdown);
    return ast;
  }

  render(ast) {
    return this.processor.stringify(ast);
  }

  parseAndRender(markdown) {
    const ast = this.parse(markdown);
    return this.render(ast);
  }
}

export function createParser(options = {}) {
  return new MarkdownParser(options);
}

export default MarkdownParser;
