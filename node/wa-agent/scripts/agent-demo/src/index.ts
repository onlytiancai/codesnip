import Anthropic from '@anthropic-ai/sdk';
import { readFile, writeFile, mkdir, readdir, stat } from 'fs/promises';
import * as readline from 'readline';
import * as cheerio from 'cheerio';
import { z } from 'zod';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { parseArgs } from 'util';

dotenv.config();

// ============ Configuration ============
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const memoryPath = path.join(__dirname, '..', 'memory.md');

// Constants
const DEFAULT_TIMEOUT_MS = 120_000;
const MAX_RETRIES = 3;
const RETRY_DELAYS = [1000, 2000, 4000];
const MAX_HISTORY_LENGTH = 100;
const MAX_TOKEN_OUTPUT = 4096;
const WEB_FETCH_MAX_CHARS = 5000;
const TOOL_RESULT_MAX_DISPLAY = 100;
const ALLOWED_BASE_DIR = process.cwd();

// Environment
const baseURL = process.env.ANTHROPIC_BASE_URL || 'https://api.anthropic.com';
const apiKey = process.env.ANTHROPIC_API_KEY || '';
const model = process.env.ANTHROPIC_MODEL || 'claude-3-5-sonnet-20241022';
const requestTimeout = parseInt(process.env.ANTHROPIC_TIMEOUT_MS || String(DEFAULT_TIMEOUT_MS), 10);

if (!apiKey) {
  console.error('Error: ANTHROPIC_API_KEY is not set');
  process.exit(1);
}

console.log(`[Config] API: ${baseURL}`);
console.log(`[Config] Model: ${model}`);
console.log(`[Config] Timeout: ${requestTimeout}ms`);

const anthropic = new Anthropic({ baseURL, apiKey });

// ============ Metrics ============
interface Metrics {
  requestCount: number;
  totalTokens: number;
  toolCalls: number;
  errors: number;
  startTime: number;
}

const metrics: Metrics = {
  requestCount: 0,
  totalTokens: 0,
  toolCalls: 0,
  errors: 0,
  startTime: Date.now(),
};

function recordRequest(tokens: number = 0, toolCallCount: number = 0): void {
  metrics.requestCount++;
  metrics.totalTokens += tokens;
  metrics.toolCalls += toolCallCount;
}

function recordError(): void {
  metrics.errors++;
}

function printMetrics(): void {
  const duration = ((Date.now() - metrics.startTime) / 1000).toFixed(2);
  console.log(`\n[Metrics] Requests: ${metrics.requestCount}, Tokens: ${metrics.totalTokens}, ToolCalls: ${metrics.toolCalls}, Errors: ${metrics.errors}, Duration: ${duration}s`);
}

// ============ Types ============
type Role = 'user' | 'assistant';

interface Message {
  role: Role;
  content: string | ContentBlock[];
}

type ContentBlock =
  | { type: 'tool_use'; id: string; name: string; input: Record<string, unknown> }
  | { type: 'tool_result'; tool_use_id: string; content: string };

interface ToolCall {
  name: string;
  id: string;
  args: Record<string, unknown>;
}

interface ToolDefinition {
  name: string;
  description: string;
  input_schema: z.ZodRawShape;
}

interface StreamResult {
  textContent: string;
  toolCalls: ToolCall[];
  usage?: { input_tokens: number; output_tokens: number };
}

// ============ Path Validation ============
function validatePath(filePath: string): string {
  const resolved = path.resolve(filePath);
  if (!resolved.startsWith(ALLOWED_BASE_DIR)) {
    throw new Error(`Path ${filePath} is outside allowed directory`);
  }
  return resolved;
}

// ============ Tools ============
const tools: Record<string, {
  description: string;
  schema: z.ZodRawShape;
  execute: (args: Record<string, unknown>) => Promise<string>;
}> = {
  read: {
    description: 'Read the contents of a file',
    schema: { path: z.string().describe('The file path to read') },
    async execute({ path: filePath }) {
      try {
        validatePath(String(filePath));
        return await readFile(String(filePath), 'utf-8');
      } catch (error) {
        return formatError(error);
      }
    },
  },
  write: {
    description: 'Write content to a file',
    schema: {
      path: z.string().describe('The file path to write'),
      content: z.string().describe('The content to write'),
    },
    async execute({ path: filePath, content }) {
      try {
        const resolved = validatePath(String(filePath));
        await mkdir(path.dirname(resolved), { recursive: true });
        await writeFile(resolved, String(content), 'utf-8');
        return `Successfully wrote to ${resolved}`;
      } catch (error) {
        return formatError(error);
      }
    },
  },
  ls: {
    description: 'List directory contents',
    schema: { path: z.string().describe('The directory path to list') },
    async execute({ path: dirPath }) {
      try {
        const resolved = validatePath(String(dirPath));
        const entries = await readdir(resolved);
        const result = await Promise.all(
          entries.map(async (name) => {
            const fullPath = path.join(resolved, name);
            const s = await stat(fullPath);
            return `${s.isDirectory() ? 'd' : '-'} ${name}`;
          })
        );
        return result.join('\n');
      } catch (error) {
        return formatError(error);
      }
    },
  },
  mkdir: {
    description: 'Create a directory',
    schema: { path: z.string().describe('The directory path to create') },
    async execute({ path: dirPath }) {
      try {
        const resolved = validatePath(String(dirPath));
        await mkdir(resolved, { recursive: true });
        return `Successfully created directory: ${resolved}`;
      } catch (error) {
        return formatError(error);
      }
    },
  },
  web_fetch: {
    description: 'Fetch content from a web URL',
    schema: { url: z.string().url().describe('The URL to fetch') },
    async execute({ url }) {
      try {
        const response = await fetch(String(url), {
          headers: { 'User-Agent': 'AI-Agent/1.0' },
          signal: AbortSignal.timeout(30000),
        });
        const html = await response.text();
        const text = cheerio.load(html)('body').text().trim();
        return truncateText(text, WEB_FETCH_MAX_CHARS);
      } catch (error) {
        return formatError(error);
      }
    },
  },
};

const toolDefinitions: ToolDefinition[] = Object.entries(tools).map(([name, tool]) => ({
  name,
  description: tool.description,
  input_schema: tool.schema,
}));

// ============ Error Handling ============
function formatError(error: unknown): string {
  if (error instanceof Error) {
    return `Error: ${error.message}`;
  }
  return `Error: ${String(error)}`;
}

function parseAPIError(error: unknown): { message: string; isRetryable: boolean } {
  if (error instanceof Error) {
    const msg = error.message;
    if (msg.includes('429') || msg.includes('rate limit')) return { message: 'Rate limit, retrying...', isRetryable: true };
    if (msg.includes('500') || msg.includes('502') || msg.includes('503')) return { message: 'Server error, retrying...', isRetryable: true };
    if (msg.includes('network') || msg.includes('ECONNREFUSED')) return { message: 'Network error, retrying...', isRetryable: true };
    if (msg.includes('timeout') || msg.includes('Timeout')) return { message: 'Request timeout, retrying...', isRetryable: true };
    return { message: msg, isRetryable: false };
  }
  return { message: String(error), isRetryable: false };
}

// ============ Text Utilities ============
function truncateText(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.substring(0, maxLen) + `... (${text.length} chars total)`;
}

function escapeNewlines(text: string): string {
  return text.replace(/\n/g, '\\n');
}

function formatToolResult(result: string, maxLen: number = TOOL_RESULT_MAX_DISPLAY): string {
  const escaped = escapeNewlines(result);
  if (escaped.length <= maxLen) return escaped;
  return truncateText(escaped, maxLen);
}

// ============ Memory ============
async function loadMemory(): Promise<string> {
  try {
    return await readFile(memoryPath, 'utf-8');
  } catch {
    return '# Agent Memory\n\nNo persistent memory available.';
  }
}

// ============ History Management ============
const history: Message[] = [];

function addToHistory(message: Message): void;
function addToHistory(...messages: Message[]): void;
function addToHistory(...messages: Message[]): void {
  const toAdd = messages.length === 1 && Array.isArray(messages[0]) ? messages[0] : messages;
  history.push(...toAdd);
  while (history.length > MAX_HISTORY_LENGTH) {
    history.shift();
  }
}

function clearHistory(): void {
  history.length = 0;
}

// ============ CLI Helpers ============
let rl: readline.Interface | null = null;

function getReadline(): readline.Interface {
  if (!rl) {
    rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  }
  return rl;
}

function closeReadline(): void {
  if (rl) {
    rl.close();
    rl = null;
  }
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const clearLine = () => process.stdout.write('\r\x1b[K');

const showStatus = (status: string) => {
  clearLine();
  process.stdout.write(`[${new Date().toLocaleTimeString()}] ${status}`);
};

// ============ Tool Execution ============
async function executeTool(name: string, args: Record<string, unknown>): Promise<string> {
  const tool = tools[name];
  if (!tool) return `Unknown tool: ${name}`;
  return await tool.execute(args);
}

// ============ Streaming ============
async function streamMessages(
  messages: Message[],
  systemPrompt: string,
): Promise<StreamResult> {
  const toolCalls: { name: string; id: string; argsRaw: string }[] = [];
  let currentToolIndex = -1;
  let textContent = '';
  let currentBlockType: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;

  const stream = anthropic.messages.stream({
    model,
    max_tokens: MAX_TOKEN_OUTPUT,
    system: systemPrompt,
    messages: messages as Anthropic.MessageParam[],
    tools: toolDefinitions as unknown as Anthropic.Tool[],
  });

  try {
    for await (const event of stream) {
      const e = event as unknown as Record<string, unknown>;

      if (e.type === 'message_start') {
        const msgStart = e as { type: string; usage?: { input_tokens: number } };
        if (msgStart.usage) {
          inputTokens = msgStart.usage.input_tokens;
        }
      } else if (e.type === 'message_delta') {
        const msgDelta = e as { type: string; usage?: { output_tokens: number }; delta?: { text?: string } };
        if (msgDelta.usage) {
          outputTokens = msgDelta.usage.output_tokens;
        }
        if (msgDelta.delta?.text) {
          process.stdout.write(msgDelta.delta.text);
          textContent += msgDelta.delta.text;
        }
      } else if (e.type === 'content_block_start') {
        const block = e as { content_block?: { type: string; name?: string; id?: string } };
        if (block.content_block?.type === 'thinking') {
          currentBlockType = 'thinking';
          process.stdout.write('[Thinking]\n');
        } else if (block.content_block?.type === 'text') {
          currentBlockType = 'text';
          process.stdout.write('[Text]\n');
        } else if (block.content_block?.type === 'tool_use') {
          currentToolIndex++;
          toolCalls.push({
            name: block.content_block.name || '',
            id: block.content_block.id || '',
            argsRaw: '',
          });
          currentBlockType = 'tool_use';
          process.stdout.write(`[Tool] ${block.content_block.name}\n`);
        }
      } else if (e.type === 'content_block_delta') {
        const delta = e as { delta?: { type: string; thinking?: string; text?: string; partial_json?: string } };
        if (delta.delta?.type === 'thinking_delta') {
          process.stdout.write(delta.delta.thinking || '');
        } else if (delta.delta?.type === 'text_delta' && delta.delta?.text) {
          process.stdout.write(delta.delta.text);
          textContent += delta.delta.text;
        } else if (delta.delta?.type === 'input_json_delta') {
          if (currentToolIndex >= 0 && toolCalls[currentToolIndex]) {
            toolCalls[currentToolIndex].argsRaw += delta.delta.partial_json || '';
          }
        }
      } else if (e.type === 'content_block_stop') {
        if (currentBlockType === 'thinking') {
          process.stdout.write('\n[/Thinking]\n');
        } else if (currentBlockType === 'text') {
          process.stdout.write('\n[/Text]\n');
        } else if (currentBlockType === 'tool_use') {
          const argsRaw = toolCalls[currentToolIndex]?.argsRaw || '';
          process.stdout.write(`${formatToolResult(argsRaw)}\n`);
          process.stdout.write('[/Tool]\n');
        }
        currentBlockType = null;
      } else if (e.type === 'message_stop') {
        console.log();
      }
    }
  } catch (error) {
    stream.controller.abort();
    throw error;
  }

  // Parse tool args with error handling
  const parsedToolCalls: ToolCall[] = toolCalls.map(tc => {
    try {
      return {
        name: tc.name,
        id: tc.id,
        args: JSON.parse(tc.argsRaw) as Record<string, unknown>,
      };
    } catch {
      return {
        name: tc.name,
        id: tc.id,
        args: {} as Record<string, unknown>,
      };
    }
  });

  recordRequest(inputTokens, parsedToolCalls.length);

  return { textContent, toolCalls: parsedToolCalls, usage: { input_tokens: inputTokens, output_tokens: outputTokens } };
}

// ============ Interactive Mode ============
async function interactiveMode(systemPrompt: string): Promise<void> {
  console.log('\n=== AI Agent Demo ===');
  console.log('Tools: read, write, ls, mkdir, web_fetch');
  console.log('Commands: /new (clear history), /exit (quit)\n');

  const input = getReadline();

  while (true) {
    const userInput = await new Promise<string>(resolve => input.question('\n> ', resolve));
    if (!userInput.trim()) continue;

    // Slash commands
    if (userInput === '/new') { clearHistory(); console.log('[New conversation]\n'); continue; }
    if (userInput === '/exit') { closeReadline(); return; }

    await processUserInput(userInput, systemPrompt);
  }
}

// ============ Non-Interactive Mode ============
async function nonInteractiveMode(prompt: string, systemPrompt: string): Promise<void> {
  console.log(`[Single-shot Mode] Executing: "${prompt}"\n`);
  await processUserInput(prompt, systemPrompt);
  closeReadline();
}

// ============ Process User Input ============
async function processUserInput(input: string, systemPrompt: string): Promise<void> {
  let success = false;
  let lastError = '';

  for (let attempt = 0; attempt <= MAX_RETRIES && !success; attempt++) {
    if (attempt > 0) {
      showStatus(`Retrying (${attempt}/${MAX_RETRIES})`);
      console.log(`\n[Retry ${attempt}] ${lastError}`);
      await sleep(RETRY_DELAYS[attempt - 1] || 4000);
    }

    try {
      showStatus('Sending request...');
      console.log('\n[LLM Response]\n');

      const { textContent, toolCalls } = await streamMessages(
        [...history, { role: 'user', content: input }],
        systemPrompt
      );

      // No tools - simple response
      if (toolCalls.length === 0) {
        addToHistory({ role: 'user', content: input });
        addToHistory({ role: 'assistant', content: textContent });
        success = true;
        showStatus('Done');
        console.log();
        continue;
      }

      // Execute tools sequentially
      console.log();
      const results: string[] = [];
      for (const tc of toolCalls) {
        process.stdout.write(`[Calling ${tc.name}]\n`);
        const result = await executeTool(tc.name, tc.args);
        results.push(result);
        console.log(`[${tc.name} result]\n${formatToolResult(result)}\n`);
      }

      // Build messages for follow-up
      const assistantMsg: Message = {
        role: 'assistant',
        content: toolCalls.map((tc) => ({
          type: 'tool_use' as const,
          id: tc.id,
          name: tc.name,
          input: tc.args,
        })),
      };

      const toolResultMsgs: Message[] = toolCalls.map((tc, i) => ({
        role: 'user' as const,
        content: [{ type: 'tool_result' as const, tool_use_id: tc.id, content: results[i] }],
      }));

      // Follow-up request with full history
      showStatus('Sending tool results...');
      console.log('\n[LLM Response]\n');

      const { textContent: finalText, toolCalls: followUpTools } = await streamMessages(
        [...history, { role: 'user', content: input }, assistantMsg, ...toolResultMsgs],
        systemPrompt
      );

      // Update history
      addToHistory({ role: 'user', content: input });
      addToHistory(assistantMsg);
      addToHistory(...toolResultMsgs);

      // Handle follow-up tools sequentially
      if (followUpTools.length > 0) {
        addToHistory({
          role: 'assistant',
          content: followUpTools.map((tc) => ({
            type: 'tool_use' as const,
            id: tc.id,
            name: tc.name,
            input: tc.args,
          })),
        });
        for (const tc of followUpTools) {
          process.stdout.write(`[Calling ${tc.name}]\n`);
          const result = await executeTool(tc.name, tc.args);
          console.log(`[${tc.name} result]\n${formatToolResult(result)}\n`);
          addToHistory({
            role: 'user' as const,
            content: [{ type: 'tool_result' as const, tool_use_id: tc.id, content: result }],
          });
        }
      }

      addToHistory({ role: 'assistant', content: finalText || textContent });
      success = true;
      showStatus('Done');
      console.log();

    } catch (error) {
      recordError();
      const { message, isRetryable } = parseAPIError(error);
      lastError = message;
      if (!isRetryable || attempt >= MAX_RETRIES) {
        console.error(`\n[Error] ${message}\n`);
        break;
      }
    }
  }
}

// ============ Main ============
async function main(): Promise<void> {
  const systemPrompt = await loadMemory();

  // Parse command line arguments
  const { values, positionals } = parseArgs({
    args: process.argv.slice(2),
    options: {
      help: { type: 'boolean', default: false, short: 'h' },
    },
    allowPositionals: true,
  });

  if (values.help) {
    console.log(`
Usage: agent-demo [OPTIONS] [PROMPT]

Options:
  -h, --help     Show this help message

Arguments:
  PROMPT         User prompt to execute in single-shot mode (non-interactive)

If PROMPT is provided, runs in non-interactive mode and exits after execution.
Otherwise, runs in interactive REPL mode.

Examples:
  agent-demo "List files in current directory"
  agent-demo --help
`);
    closeReadline();
    return;
  }

  const prompt = positionals.join(' ');

  if (prompt) {
    await nonInteractiveMode(prompt, systemPrompt);
  } else {
    await interactiveMode(systemPrompt);
  }

  printMetrics();
}

process.on('SIGINT', () => {
  console.log('\n\nExiting...');
  closeReadline();
  printMetrics();
  process.exit(0);
});

process.on('exit', () => {
  closeReadline();
});

main().catch((error) => {
  console.error('Fatal error:', error);
  closeReadline();
  printMetrics();
  process.exit(1);
});
