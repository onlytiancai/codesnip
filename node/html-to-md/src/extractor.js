const { parse } = require('node-html-parser');
const { extractFromHtml } = require('@extractus/article-extractor');

/**
 * 使用 @mozilla/readability 提取内容
 * @param {string} html - HTML 内容
 * @returns {{title: string, content: string}}
 */
function extractWithReadability(html) {
  const { JSDOM } = require('jsdom');
  try {
    const dom = new JSDOM(html);
    const { Readability } = require('@mozilla/readability');
    const reader = new Readability(dom.window.document);
    const result = reader.parse();

    if (result && result.content) {
      return {
        title: result.title || 'Untitled',
        content: result.content,
      };
    }
  } catch (error) {
    throw new Error(`Readability failed: ${error.message}`);
  }

  return null;
}

/**
 * 提取 HTML 中的主体内容
 * @param {string} html - HTML 内容
 * @param {Object} options - 选项
 * @param {boolean} options.useReadability - 是否使用 Readability
 * @returns {Promise<{title: string, content: string}>} 标题和内容
 */
async function extractContent(html, options = {}) {
  const { useReadability = false } = options;

  // 如果指定使用 Readability
  if (useReadability) {
    console.log('Using @mozilla/readability for extraction...');
    const result = extractWithReadability(html);
    if (result) {
      return result;
    }
  }

  // 首选：使用 article-extractor 智能提取
  try {
    const article = await extractFromHtml(html);
    if (article && article.content) {
      return {
        title: article.title || 'Untitled',
        content: article.content,
      };
    }
  } catch (error) {
    console.warn('Article extractor failed, using fallback method:', error.message);
  }

  // 备选：手动提取
  return extractContentFallback(html);
}

/**
 * 备选的内容提取方法
 * @param {string} html - HTML 内容
 * @returns {{title: string, content: string}}
 */
function extractContentFallback(html) {
  const root = parse(html);

  // 提取标题
  let title = root.querySelector('title')?.textContent ||
              root.querySelector('h1')?.textContent ||
              'Untitled';
  title = title.trim();

  // 尝试按优先级选择内容区域
  const selectors = [
    'article',
    'main',
    '[role="main"]',
    '.post-content',
    '.article-content',
    '.content',
    '#content',
  ];

  let contentElement = null;
  for (const selector of selectors) {
    contentElement = root.querySelector(selector);
    if (contentElement) break;
  }

  // 如果都没找到，使用 body
  if (!contentElement) {
    contentElement = root.querySelector('body') || root;
  }

  // 移除不需要的元素
  const unwantedSelectors = [
    'script',
    'style',
    'noscript',
    'iframe',
    'nav',
    'footer',
    'header',
    '.ad',
    '.ads',
    '.advertisement',
    '.nav',
    '.footer',
    '.sidebar',
  ];

  for (const selector of unwantedSelectors) {
    const elements = contentElement.querySelectorAll(selector);
    elements.forEach(el => el.remove());
  }

  return {
    title,
    content: contentElement.innerHTML,
  };
}

module.exports = { extractContent, extractContentFallback };
