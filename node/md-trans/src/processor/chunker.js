import { NodeType, classifyNode } from '../parser/nodeClassifier.js';

const DEFAULT_MAX_TOKENS = 2000;

export class Chunker {
  constructor(options = {}) {
    this.maxTokens = options.maxTokens || DEFAULT_MAX_TOKENS;
    this.estimateTokens = options.estimateTokens || this._estimateTokens;
  }

  _estimateTokens(text) {
    return Math.ceil(text.length / 4);
  }

  chunk(ast) {
    const chunks = [];
    this._traverseAndChunk(ast, chunks, []);
    return chunks;
  }

  _traverseAndChunk(node, chunks, currentChunk, path = []) {
    if (!node || typeof node !== 'object') {
      return;
    }

    if (currentChunk.length === 0) {
      currentChunk.push({ node, path });
    } else {
      const lastChunk = chunks[chunks.length - 1];
      const estimatedTokens = this._estimateChunkTokens(lastChunk);

      if (estimatedTokens < this.maxTokens) {
        lastChunk.push({ node, path });
      } else {
        const newChunk = [{ node, path }];
        chunks.push(newChunk);
      }
    }

    if (node.children && Array.isArray(node.children)) {
      const childChunks = [];
      for (let i = 0; i < node.children.length; i++) {
        const child = node.children[i];
        const childPath = [...path, { type: node.type, index: i }];

        if (this._shouldStartNewChunk(child)) {
          if (childChunks.length > 0) {
            const combined = this._combineChunks(childChunks);
            const lastChunk = chunks[chunks.length - 1];
            if (this._estimateChunkTokens(lastChunk) + this._estimateChunkTokens(combined) < this.maxTokens) {
              lastChunk.push(...combined);
            } else {
              chunks.push(combined);
            }
            childChunks.length = 0;
          }
          chunks.push([{ node: child, path: childPath }]);
        } else {
          childChunks.push({ node: child, path: childPath });
        }
      }

      if (childChunks.length > 0) {
        const lastChunk = chunks[chunks.length - 1];
        const combined = this._combineChunks(childChunks);
        if (this._estimateChunkTokens(lastChunk) + this._estimateChunkTokens(combined) < this.maxTokens) {
          lastChunk.push(...combined);
        } else {
          chunks.push(combined);
        }
      }
    }
  }

  _shouldStartNewChunk(node) {
    if (!node || !node.type) return false;

    const blockTypes = ['heading', 'blockquote', 'list', 'orderedList', 'unorderedList', 'table'];
    return blockTypes.includes(node.type);
  }

  _combineChunks(chunks) {
    return chunks;
  }

  _estimateChunkTokens(chunk) {
    if (!Array.isArray(chunk)) {
      return this.estimateTokens(this._nodeToText(chunk));
    }

    let totalTokens = 0;
    for (const item of chunk) {
      totalTokens += this.estimateTokens(this._nodeToText(item.node));
    }
    return totalTokens;
  }

  _nodeToText(node) {
    if (!node) return '';

    if (node.type === 'text') {
      return node.value || '';
    }

    if (node.type === 'heading') {
      return (node.children || []).map(c => this._nodeToText(c)).join('');
    }

    if (node.type === 'paragraph') {
      return (node.children || []).map(c => this._nodeToText(c)).join('');
    }

    if (node.type === 'inlineCode') {
      return node.value || '';
    }

    if (node.children) {
      return node.children.map(c => this._nodeToText(c)).join(' ');
    }

    return '';
  }

  chunkByNode(ast) {
    const nodes = [];
    this._collectNodes(ast, nodes);
    return nodes.map(node => ({ node, path: [] }));
  }

  _collectNodes(node, nodes, path = []) {
    if (!node || typeof node !== 'object') {
      return;
    }

    const nodeType = classifyNode(node);
    if (nodeType !== NodeType.SKIP) {
      nodes.push(node);
    }

    if (node.children && Array.isArray(node.children)) {
      for (let i = 0; i < node.children.length; i++) {
        this._collectNodes(node.children[i], nodes, [...path, { type: node.type, index: i }]);
      }
    }
  }
}

export function createChunker(options = {}) {
  return new Chunker(options);
}

export default Chunker;
