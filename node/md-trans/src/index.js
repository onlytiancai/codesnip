import { createParser } from './parser/markdownParser.js';
import { createTraverser } from './parser/astTraverser.js';
import { createPreAnalyzer } from './translator/preAnalyzer.js';
import { createTranslator } from './translator/translator.js';
import { createClient } from './translator/llmClient.js';
import { createCache } from './processor/consistencyCache.js';
import { createRenderer } from './processor/bilingualRenderer.js';
import { createLogger } from './utils/logger.js';
import { readMarkdown, writeMarkdown } from './utils/fileHandler.js';

export class MDTrans {
  constructor(options = {}) {
    this.options = options;
    this.logger = createLogger({ debug: options.debug });
    this.parser = createParser();
    this.traverser = createTraverser();
    this.renderer = createRenderer({ logger: this.logger });

    this.client = createClient({
      mode: options.llmMode,
      apiKey: options.apiKey,
      baseURL: options.baseURL,
      model: options.model,
      timeout: (options.timeout || 120) * 1000,
      logger: this.logger,
    });

    this.cache = createCache({ logger: this.logger });
    this.preAnalyzer = createPreAnalyzer({ client: this.client, logger: this.logger });
    this.translator = createTranslator({
      client: this.client,
      cache: this.cache,
      logger: this.logger,
      template: options.template,
      skipPreanalysis: options.skipPreanalysis,
    });
  }

  async translateFile(inputPath, outputPath, options = {}) {
    this.logger.info(`Reading input file: ${inputPath}`);
    const content = await readMarkdown(inputPath);
    const result = await this.translate(content, options);

    if (outputPath) {
      this.logger.info(`Writing output file: ${outputPath}`);
      await writeMarkdown(outputPath, result);
    }

    return result;
  }

  async translate(markdown, options = {}) {
    this.logger.info('Parsing markdown...');
    const ast = this.parser.parse(markdown);

    let preanalysis = null;

    if (!this.options.skipPreanalysis && !options.skipPreanalysis) {
      this.logger.info('Stage 1: Pre-analysis...');
      const textContent = this._extractAllText(ast);
      preanalysis = await this.preAnalyzer.analyze(textContent);
      this.cache.loadFromPreAnalysis(preanalysis);
      this.logger.debug('Pre-analysis completed', preanalysis);
    }

    this.logger.info('Stage 2: Translation...');
    const translationResults = await this.translator.translate(ast, preanalysis, options);

    this.logger.info('Rendering bilingual output...');
    const bilingualOutput = this._renderBilingual(ast, translationResults);

    return bilingualOutput;
  }

  _extractAllText(ast) {
    const texts = [];
    this.traverser.traverse(ast, (node) => {
      if (node.type === 'text') {
        texts.push(node.value);
      }
    });
    return texts.join('\n\n');
  }

  _renderBilingual(ast, translationResults) {
    const lines = [];
    const resultMap = new Map();

    for (const result of translationResults) {
      if (result.node && result.node.position) {
        const key = `${result.node.position.start.line}-${result.node.position.end.line}`;
        resultMap.set(key, result);
      } else {
        resultMap.set(result.node?.type, result);
      }
    }

    this._traverseAndRender(ast, lines, resultMap, 0);

    return lines.join('\n');
  }

  _traverseAndRender(node, lines, resultMap, depth) {
    if (!node || typeof node !== 'object') {
      return depth;
    }

    const nodeType = node.type;

    if (nodeType === 'code' || nodeType === 'inlineCode') {
      lines.push(this.renderer._getNodeText(node));
      return depth;
    }

    if (nodeType === 'text' && node.parent && node.parent.type === 'paragraph') {
      return depth;
    }

    if (nodeType === 'heading') {
      const origText = this.renderer._getNodeText(node);
      const result = resultMap.get(node) || resultMap.get(nodeType);
      const transText = result?.translation || '';

      lines.push(`${'#'.repeat(node.depth)} ${origText}`);
      lines.push(`${'#'.repeat(node.depth)} ${transText}`);
      lines.push('');
      return depth;
    }

    if (nodeType === 'paragraph') {
      const origText = this.renderer._getNodeText(node);
      const result = resultMap.get(nodeType);
      const transText = result?.translation || '';

      if (origText.trim()) {
        lines.push(origText);
        lines.push(transText);
        lines.push('');
      }
      return depth;
    }

    if (nodeType === 'blockquote') {
      const origText = this.renderer._getNodeText(node);
      const result = resultMap.get(nodeType);
      const transText = result?.translation || '';

      const origLines = origText.split('\n');
      for (const line of origLines) {
        if (line.trim()) lines.push(`> ${line}`);
      }
      lines.push('');

      const transLines = transText.split('\n');
      for (const line of transLines) {
        if (line.trim()) lines.push(`> ${line}`);
      }
      lines.push('');
      return depth;
    }

    if (nodeType === 'list' || nodeType === 'orderedList' || nodeType === 'unorderedList') {
      const result = resultMap.get(nodeType);
      const transText = result?.translation || '';

      if (transText) {
        lines.push(transText);
        lines.push('');
      }
      return depth;
    }

    if (nodeType === 'code') {
      const lang = node.lang || '';
      const code = node.value || '';
      lines.push(`\`\`\`${lang}`);
      lines.push(code);
      lines.push('```');
      lines.push('');
      return depth;
    }

    if (nodeType === 'thematicBreak') {
      lines.push('---');
      lines.push('');
      return depth;
    }

    if (node.children && Array.isArray(node.children)) {
      for (const child of node.children) {
        depth = this._traverseAndRender(child, lines, resultMap, depth);
      }
    }

    return depth;
  }
}

export async function createMDTrans(options = {}) {
  return new MDTrans(options);
}

export async function translate(input, output, options = {}) {
  const mdTrans = new MDTrans(options);
  return mdTrans.translateFile(input, output, options);
}

export default MDTrans;
