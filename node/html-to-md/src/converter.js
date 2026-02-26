const TurndownService = require('turndown');
const { gfm } = require('turndown-plugin-gfm');

/**
 * 创建 Turndown 服务实例，配置自定义规则
 * @returns {TurndownService}
 */
function createTurndownService() {
  const turndownService = new TurndownService({
    headingStyle: 'atx',
    codeBlockStyle: 'fenced',
    fence: '```',
    emDelimiter: '*',
    bulletListMarker: '-',
  });

  // 使用 GFM 插件（支持表格、任务列表等）
  turndownService.use(gfm);

  // 自定义代码块规则
  turndownService.addRule('codeBlock', {
    filter: ['pre'],
    replacement: function (content, node) {
      const code = node.querySelector('code');
      if (code) {
        const className = code.className || '';
        const match = className.match(/language-(\w+)/);
        const language = match ? match[1] : '';
        return `\n\`\`\`${language}\n${code.textContent.trim()}\n\`\`\`\n`;
      }
      return `\n\`\`\`\n${content.trim()}\n\`\`\`\n`;
    },
  });

  // 处理 <br> 标签
  turndownService.addRule('br', {
    filter: 'br',
    replacement: function () {
      return '  \n';
    },
  });

  return turndownService;
}

/**
 * 将 HTML 转换为 Markdown
 * @param {string} html - HTML 内容
 * @returns {string} Markdown 内容
 */
function convertToMarkdown(html) {
  const turndownService = createTurndownService();
  return turndownService.turndown(html);
}

module.exports = { convertToMarkdown, createTurndownService };
