#!/usr/bin/env node

const { program } = require('commander');
const { readFile, writeFile } = require('fs/promises');
const { fetchHtml } = require('./fetcher');
const { extractContent } = require('./extractor');
const { convertToMarkdown } = require('./converter');

/**
 * 主处理函数
 * @param {Object} options - 选项
 * @param {string} options.url - URL
 * @param {string} options.file - 本地文件路径
 * @param {string} options.output - 输出文件路径
 * @param {boolean} options.readability - 使用 Readability 提取
 */
async function processHtmlToMd(options) {
  let html = '';

  // 确定输入源
  if (options.url) {
    console.log(`Fetching HTML from: ${options.url}`);
    html = await fetchHtml(options.url);
  } else if (options.file) {
    console.log(`Reading HTML from: ${options.file}`);
    try {
      html = await readFile(options.file, 'utf-8');
    } catch (error) {
      if (error.code === 'ENOENT') {
        throw new Error(`File not found: ${options.file}`);
      }
      throw error;
    }
  } else {
    throw new Error('Please provide either a URL (-u) or a file (-f)');
  }

  // 提取内容
  console.log('Extracting content...');
  const { title, content } = await extractContent(html, { useReadability: options.readability });

  // 转换为 Markdown
  console.log('Converting to Markdown...');
  let markdown = `# ${title}\n\n`;
  markdown += convertToMarkdown(content);

  // 输出
  if (options.output) {
    console.log(`Writing to: ${options.output}`);
    await writeFile(options.output, markdown, 'utf-8');
    console.log('Done!');
  } else {
    // 输出到 stdout
    console.log('\n--- Markdown Output ---\n');
    console.log(markdown);
  }
}

// 交互式模式
async function interactiveMode() {
  console.log('HTML to Markdown Converter (Interactive Mode)');
  console.log('---------------------------------------------');

  const { createInterface } = require('readline');
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const question = (prompt) => new Promise((resolve) => rl.question(prompt, resolve));

  try {
    const choice = await question('Input source (url/file): ');

    let url, file;
    if (choice.toLowerCase().startsWith('u')) {
      url = await question('Enter URL: ');
    } else {
      file = await question('Enter file path: ');
    }

    const output = await question('Output file path (optional, press enter to print to stdout): ');

    await processHtmlToMd({ url, file, output: output || undefined });
  } finally {
    rl.close();
  }
}

// CLI 配置
program
  .name('html-to-md')
  .description('Convert HTML to Markdown')
  .version('1.0.0')
  .allowExcessArguments(false)
  .allowUnknownOption(false);

program
  .option('-u, --url <url>', 'URL to fetch HTML from')
  .option('-f, --file <path>', 'Local HTML file path')
  .option('-o, --output <path>', 'Output Markdown file path')
  .option('-r, --readability', 'Use @mozilla/readability for content extraction')
  .action(async (options) => {
    try {
      if (!options.url && !options.file) {
        await interactiveMode();
      } else {
        await processHtmlToMd(options);
      }
    } catch (error) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  });

program.parse();
