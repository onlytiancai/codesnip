import { isSkip } from './nodeClassifier.js';

export class ASTTraverser {
  constructor(options = {}) {
    this.options = options;
  }

  traverse(ast, callback) {
    this._traverseNode(ast, callback, []);
  }

  _traverseNode(node, callback, path) {
    if (!node || typeof node !== 'object') {
      return;
    }

    const nodePath = [...path];
    callback(node, nodePath);

    if (node.children && Array.isArray(node.children)) {
      for (const child of node.children) {
        this._traverseNode(child, callback, [...nodePath, node]);
      }
    }
  }

  traverseWithContext(ast, callback) {
    this._traverseWithContextRecursive(ast, callback, {
      parent: null,
      path: [],
      index: 0,
      siblings: [],
    });
  }

  _traverseWithContextRecursive(node, callback, context) {
    if (!node || typeof node !== 'object') {
      return;
    }

    callback(node, context);

    if (node.children && Array.isArray(node.children)) {
      const children = node.children;
      for (let i = 0; i < children.length; i++) {
        this._traverseWithContextRecursive(children[i], callback, {
          parent: node,
          path: [...context.path, node],
          index: i,
          siblings: children,
        });
      }
    }
  }

  getTranslatableNodes(ast) {
    const nodes = [];
    this.traverse(ast, (node, path) => {
      if (!isSkip(node)) {
        nodes.push({ node, path });
      }
    });
    return nodes;
  }
}

export function createTraverser(options = {}) {
  return new ASTTraverser(options);
}

export default ASTTraverser;
