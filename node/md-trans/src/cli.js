#!/usr/bin/env node

import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { config } from 'dotenv';
import { MDTrans } from './index.js';
import { createClient } from './translator/llmClient.js';
import picocolors from 'picocolors';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = resolve(__dirname, '..');

// Load .env file (override existing env vars)
config({ path: resolve(rootDir, '.env'), quiet: true, override: true });

const argv = yargs(process.argv.slice(2))
  .usage('Usage: md-trans <input> [-o <output>] [options]')
  .positional('input', {
    describe: 'Input markdown file',
    type: 'string',
  })
  .option('output', {
    alias: 'o',
    describe: 'Output markdown file',
    type: 'string',
  })
  .option('template', {
    alias: 't',
    describe: 'Custom prompt template file',
    type: 'string',
  })
  .option('debug', {
    alias: 'd',
    describe: 'Enable debug mode',
    type: 'boolean',
    default: false,
  })
  .option('skip-preanalysis', {
    describe: 'Skip pre-analysis stage (faster but less consistent)',
    type: 'boolean',
    default: false,
  })
  .option('chunk-size', {
    describe: 'Maximum chunk size in tokens',
    type: 'number',
    default: 2000,
  })
  .option('timeout', {
    describe: 'Request timeout in seconds',
    type: 'number',
    default: 120,
  })
  .option('api-key', {
    describe: 'OpenAI API key',
    type: 'string',
  })
  .option('base-url', {
    describe: 'OpenAI API base URL',
    type: 'string',
  })
  .option('model', {
    describe: 'OpenAI model name',
    type: 'string',
  })
  .option('llm-mode', {
    describe: 'LLM mode: anthropic or openai',
    type: 'string',
    default: 'anthropic',
    choices: ['anthropic', 'openai'],
  })
  .option('test-llm', {
    describe: 'Test LLM connection by sending "你好"',
    type: 'boolean',
    default: false,
  })
  .help()
  .alias('help', 'h')
  .version('0.1.0')
  .alias('version', 'v')
  .epilog('Environment variables: ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL')
  .argv;

async function main() {
  const { _, output, template, debug, skipPreanalysis, chunkSize, timeout, llmMode, apiKey, baseUrl, model, testLlm } = argv;
  const input = _[0];

  console.log(picocolors.cyan('md-trans - Markdown Bilingual Translation CLI'));
  console.log(picocolors.gray('='.repeat(50)));

  if (debug) {
    console.log(picocolors.gray('Debug mode enabled'));
  }

  const apiKeyValue = apiKey || process.env.OPENAI_API_KEY;
  const baseURL = baseUrl || process.env.OPENAI_BASE_URL;
  const modelValue = model || process.env.OPENAI_MODEL;

  if (!apiKeyValue) {
    console.error(picocolors.red('Error: OPENAI_API_KEY is required'));
    console.error(picocolors.gray('Set it via --api-key, .env file, or export OPENAI_API_KEY=your_key'));
    process.exit(1);
  }

  // Test LLM mode
  if (testLlm) {
    console.log(picocolors.blue('Testing LLM connection...'));
    console.log(picocolors.gray('Sending: 你好\n'));

    try {
      const client = createClient({
        apiKey: apiKeyValue,
        baseURL: baseURL,
        model: modelValue,
      });

      const response = await client.complete('你好', { temperature: 0.3 });
      console.log(picocolors.green('LLM Response:'));
      console.log(response);
      return;
    } catch (error) {
      console.error(picocolors.red('LLM Test failed:'), error.message);
      if (debug) {
        console.error(error.stack);
      }
      process.exit(1);
    }
  }

  if (!input) {
    console.error(picocolors.red('Error: Input file is required'));
    console.error(picocolors.gray('Usage: md-trans <input> [-o <output>]'));
    process.exit(1);
  }

  const options = {
    debug,
    skipPreanalysis,
    chunkSize,
    timeout,
    llmMode,
    apiKey: apiKeyValue,
    baseURL: baseURL,
    model: modelValue,
    template,
  };

  try {
    const mdTrans = new MDTrans(options);

    console.log(picocolors.blue('Starting translation...'));

    const result = await mdTrans.translateFile(input, output);

    console.log(picocolors.green('Translation completed!'));

    if (!output) {
      console.log(picocolors.gray('\n--- Output ---\n'));
      console.log(result);
    } else {
      console.log(picocolors.green(`Output saved to: ${output}`));
    }
  } catch (error) {
    const statusCode = error.response?.status;
    const statusText = statusCode ? ` [HTTP ${statusCode}]` : '';
    console.error(picocolors.red('Translation failed:'), error.message + statusText);
    if (debug) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

main();
