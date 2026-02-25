#!/usr/bin/env node

const { program } = require('commander');
const fs = require('fs').promises;
const path = require('path');
const TurndownService = require('turndown');

// 默认配置
const DEFAULT_CONFIG = {
  inputFile: '1.html',
  outputFile: '1.md',
  verbose: false,
  clean: false,
  tagsToKeep: ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br', 'ul', 'ol', 'li', 'strong', 'em', 'code', 'pre', 'blockquote', 'a', 'img'],
  tagsToRemove: ['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'input', 'button', 'iframe', 'video', 'audio']
};

// 创建 Turndown 实例
const turndownService = new TurndownService({
  headingStyle: 'atx',
  codeBlockStyle: 'fenced',
  emDelimiter: '*',
  strongDelimiter: '**',
  codeBlockDelimiter: '```'
});

// 扩展 Turndown 配置
turndownService.addRule('bold', {
  filter: ['span'],
  replacement: function(content, node) {
    if (node.style?.fontWeight === 'bold') {
      return `**${content}**`;
    }
    return content;
  }
});

turndownService.addRule('code', {
  filter: ['span'],
  replacement: function(content, node) {
    if (node.style?.fontFamily?.includes('monospace')) {
      return `\`${content}\``;
    }
    return content;
  }
});

// 显示帮助信息
function showHelp() {
  console.log('HTML to Markdown Converter');
  console.log('Converts HTML files to Markdown format');
  console.log('');
  console.log('Usage:');
  console.log('  html-to-md [options]');
  console.log('');
  console.log('Options:');
  console.log('  -i, --input <file>    Input HTML file path (default: 1.html)');
  console.log('  -o, --output <file>   Output Markdown file path (default: 1.md)');
  console.log('  -v, --verbose         Show detailed processing information');
  console.log('  -c, --clean           Only keep content tags, filter non-content tags');
  console.log('  -h, --help            Show help information');
  console.log('');
  console.log('Examples:');
  console.log('  html-to-md -i input.html -o output.md');
  console.log('  html-to-md --input article.html --output article.md --verbose');
  console.log('  html-to-md --clean --input webpage.html --output clean.md');
}

// 解析命令行参数
program
  .name('html-to-md')
  .description('Convert HTML to Markdown')
  .version('1.0.0')
  .option('-i, --input <file>', 'Input HTML file path')
  .option('-o, --output <file>', 'Output Markdown file path')
  .option('-v, --verbose', 'Show detailed processing information')
  .option('-c, --clean', 'Only keep content tags, filter non-content tags')
  .helpOption('-h, --help', 'Show help information')
  .parse();

const options = program.opts();

// 创建配置对象，优先使用用户指定的选项
const config = {
  ...DEFAULT_CONFIG,
  inputFile: options.input || DEFAULT_CONFIG.inputFile,
  outputFile: options.output || DEFAULT_CONFIG.outputFile,
  verbose: options.verbose || DEFAULT_CONFIG.verbose,
  clean: options.clean || DEFAULT_CONFIG.clean
};

// 处理 HTML 内容
async function processHTML(html, config) {
  if (config.verbose) {
    console.log('Processing HTML content...');
    console.log('HTML length:', html.length);
  }

  // 创建 DOM 解析器
  const dom = new (require('jsdom')).JSDOM(html);
  const document = dom.window.document;

  // 如果启用清理模式，处理标签
  if (config.clean) {
    if (config.verbose) {
      console.log('Applying clean mode - filtering tags...');
    }

    // 创建一个新的 div 来存储过滤后的内容
    const cleanContent = document.createElement('div');

    // 只处理 body 内容
    const bodyChildren = Array.from(document.body.childNodes);

    bodyChildren.forEach(node => {
      const tagName = node.nodeType === 1 ? node.tagName.toLowerCase() : '';

      // 如果是文本节点，直接添加
      if (node.nodeType === 3) {
        cleanContent.appendChild(node.cloneNode(true));
      }
      // 如果是元素节点且在保留列表中，添加克隆
      else if (node.nodeType === 1 && config.tagsToKeep.includes(tagName)) {
        cleanContent.appendChild(node.cloneNode(true));
      }
      // 如果是需要移除的标签，获取其文本内容
      else if (node.nodeType === 1) {
        const textContent = node.textContent || '';
        if (textContent.trim()) {
          const textNode = document.createTextNode(textContent);
          cleanContent.appendChild(textNode);
        }
      }
    });

    html = cleanContent.innerHTML;
  } else {
    // 移除不需要的标签
    if (config.tagsToRemove.length > 0) {
      if (config.verbose) {
        console.log('Removing tags:', config.tagsToRemove.join(', '));
      }

      config.tagsToRemove.forEach(tag => {
        const elements = document.querySelectorAll(tag);
        elements.forEach(element => {
          const parent = element.parentNode;
          if (parent) {
            parent.removeChild(element);
          }
        });
      });
    }

    html = document.body.innerHTML;
  }

  if (config.verbose) {
    console.log('Cleaned HTML length:', html.length);
  }

  return html;
}

// 主转换函数
async function convertHTMLtoMarkdown() {
  try {
    if (config.verbose) {
      console.log('Starting HTML to Markdown conversion...');
      console.log('Input file:', config.inputFile);
      console.log('Output file:', config.outputFile);
      console.log('Config object:', JSON.stringify(config, null, 2));
    }

    // 检查输入文件是否存在
    try {
      await fs.access(config.inputFile);
      if (config.verbose) {
        console.log('Input file found:', config.inputFile);
      }
    } catch (error) {
      console.error('Error: Input file not found:', config.inputFile);
      process.exit(1);
    }

    // 读取 HTML 文件
    if (config.verbose) {
      console.log('Reading HTML file...');
    }

    const html = await fs.readFile(config.inputFile, 'utf8');

    if (config.verbose) {
      console.log('File read successfully. Size:', html.length, 'characters');
    }

    // 处理 HTML
    const processedHTML = await processHTML(html, config);

    // 转换为 Markdown
    if (config.verbose) {
      console.log('Converting to Markdown...');
    }

    const markdown = turndownService.turndown(processedHTML);

    // 添加元数据信息
    const metadata = `<!--
  Converted from: ${config.inputFile}
  Generated on: ${new Date().toISOString()}
  Clean mode: ${config.clean}
-->\\n\\n`;

    const finalMarkdown = metadata + markdown;

    // 确保输出目录存在
    const outputDir = path.dirname(config.outputFile);
    if (outputDir !== '.') {
      await fs.mkdir(outputDir, { recursive: true });
    }

    // 写入 Markdown 文件
    await fs.writeFile(config.outputFile, finalMarkdown, 'utf8');

    if (config.verbose) {
      console.log('Markdown file saved:', config.outputFile);
      console.log('Markdown length:', finalMarkdown.length, 'characters');
    }

    console.log('✅ Conversion completed successfully!');
    console.log(`   Input: ${config.inputFile}`);
    console.log(`   Output: ${config.outputFile}`);
    console.log(`   Clean mode: ${config.clean ? 'enabled' : 'disabled'}`);

  } catch (error) {
    console.error('Error during conversion:', error.message);
    if (config.verbose) {
      console.error('Full error:', error);
    }
    process.exit(1);
  }
}

// 如果直接运行脚本而不是作为模块导入
if (require.main === module) {
  convertHTMLtoMarkdown();
}

module.exports = { convertHTMLtoMarkdown, config };