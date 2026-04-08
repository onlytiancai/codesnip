export const NodeType = {
  SKIP: 'SKIP',
  TRANSLATABLE: 'TRANSLATABLE',
  SPECIAL: 'SPECIAL',
};

export const SkipNodeTypes = [
  'code',
  'inlineCode',
  'html',
  'definition',
  'footnoteDefinition',
];

export const SpecialNodeTypes = [
  'link',
  'image',
  'linkText',
];

export function classifyNode(node) {
  if (!node || !node.type) {
    return NodeType.SKIP;
  }

  if (SkipNodeTypes.includes(node.type)) {
    return NodeType.SKIP;
  }

  if (node.type === 'code' || node.type === 'inlineCode') {
    return NodeType.SKIP;
  }

  if (node.type === 'html') {
    return NodeType.SKIP;
  }

  if (node.type === 'definition' || node.type === 'footnoteDefinition') {
    return NodeType.SKIP;
  }

  if (SpecialNodeTypes.includes(node.type)) {
    return NodeType.SPECIAL;
  }

  const translatableTypes = [
    'heading',
    'paragraph',
    'listItem',
    'tableCell',
    'blockquote',
    'text',
    'emphasis',
    'strong',
    'delete',
    'table',
    'tableRow',
    'tableHead',
    'tableBody',
    'list',
    'orderedList',
    'unorderedList',
    'item',
    'term',
    'definition',
    'description',
    'descriptionList',
    'descriptionItem',
    'break',
    'thematicBreak',
    'footnote',
    'footnoteReference',
  ];

  if (translatableTypes.includes(node.type)) {
    return NodeType.TRANSLATABLE;
  }

  return NodeType.TRANSLATABLE;
}

export function isTranslatable(node) {
  return classifyNode(node) === NodeType.TRANSLATABLE;
}

export function isSkip(node) {
  return classifyNode(node) === NodeType.SKIP;
}

export function isSpecial(node) {
  return classifyNode(node) === NodeType.SPECIAL;
}

export default {
  NodeType,
  classifyNode,
  isTranslatable,
  isSkip,
  isSpecial,
};
