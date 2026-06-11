import { NodeType, classifyNode } from '../parser/nodeClassifier.js';

export class BilingualRenderer {
  constructor(options = {}) {
    this.preserveSpacing = options.preserveSpacing !== false;
    this.logger = options.logger;
  }

  renderNode(node, translation) {
    if (!node) return '';

    switch (node.type) {
      case 'heading':
        return this._renderHeading(node, translation);
      case 'paragraph':
        return this._renderParagraph(node, translation);
      case 'text':
        return this._renderText(node, translation);
      case 'blockquote':
        return this._renderBlockquote(node, translation);
      case 'list':
      case 'orderedList':
      case 'unorderedList':
        return this._renderList(node, translation);
      case 'listItem':
        return this._renderListItem(node, translation);
      case 'table':
        return this._renderTable(node, translation);
      case 'tableRow':
        return this._renderTableRow(node, translation);
      case 'tableCell':
        return this._renderTableCell(node, translation);
      case 'code':
        return this._renderCode(node);
      case 'inlineCode':
        return this._renderInlineCode(node);
      case 'thematicBreak':
        return this._renderThematicBreak();
      case 'emphasis':
        return this._renderEmphasis(node, translation);
      case 'strong':
        return this._renderStrong(node, translation);
      default:
        return translation || this._getNodeText(node);
    }
  }

  _renderHeading(node, translation) {
    const depth = node.depth || 1;
    const prefix = '#'.repeat(depth);
    const originalText = this._getNodeText(node);
    return `${prefix} ${originalText}\n\n${prefix} ${translation}`;
  }

  _renderParagraph(node, translation) {
    const originalText = this._getNodeText(node);
    return `${originalText}\n\n${translation}`;
  }

  _renderText(node, translation) {
    return translation || node.value || '';
  }

  _renderBlockquote(node, translation) {
    const originalText = this._getNodeText(node);
    const translatedText = translation;
    const originalLines = originalText.split('\n').map(line => `> ${line}`).join('\n');
    const translatedLines = translatedText.split('\n').map(line => `> ${line}`).join('\n');
    return `${originalLines}\n\n${translatedLines}`;
  }

  _renderList(node, translation) {
    return translation;
  }

  _renderListItem(node, translation) {
    return translation;
  }

  _renderTable(node, translation) {
    return translation;
  }

  _renderTableRow(node, translation) {
    return translation;
  }

  _renderTableCell(node, translation) {
    return translation || this._getNodeText(node);
  }

  _renderCode(node) {
    const lang = node.lang || '';
    const code = node.value || '';
    return `\`\`\`${lang}\n${code}\n\`\`\``;
  }

  _renderInlineCode(node) {
    return node.value || '';
  }

  _renderThematicBreak() {
    return '---';
  }

  _renderEmphasis(node, translation) {
    return `_${translation}_`;
  }

  _renderStrong(node, translation) {
    return `**${translation}**`;
  }

  _getNodeText(node) {
    if (!node) return '';

    if (node.type === 'text') {
      return node.value || '';
    }

    if (node.children && Array.isArray(node.children)) {
      return node.children.map(child => this._getNodeText(child)).join('');
    }

    if (node.value) {
      return node.value;
    }

    return '';
  }

  renderParallel(original, translation) {
    const lines = [];

    if (original.type === 'heading') {
      const depth = original.depth || 1;
      const prefix = '#'.repeat(depth);
      const origText = this._getNodeText(original);
      lines.push(`${prefix} ${origText}`);
      lines.push(`${prefix} ${translation}`);
    } else if (original.type === 'paragraph') {
      const origText = this._getNodeText(original);
      lines.push(origText);
      lines.push(translation);
    } else if (original.type === 'blockquote') {
      const origLines = this._getNodeText(original).split('\n');
      for (const line of origLines) {
        lines.push(`> ${line}`);
      }
      const transLines = translation.split('\n');
      lines.push('');
      for (const line of transLines) {
        lines.push(`> ${line}`);
      }
    } else if (original.type === 'code') {
      const lang = original.lang || '';
      const code = original.value || '';
      lines.push(`\`\`\`${lang}`);
      lines.push(code);
      lines.push('```');
    } else {
      lines.push(this._getNodeText(original));
      lines.push(translation);
    }

    return lines.join('\n');
  }
}

export function createRenderer(options = {}) {
  return new BilingualRenderer(options);
}

export default BilingualRenderer;
