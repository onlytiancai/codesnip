import Anthropic from '@anthropic-ai/sdk';
import { readFile } from 'fs/promises';
import * as readline from 'readline';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { parseArgs } from 'util';
import {
  ToolCall,
  toolDefinitions,
  toolNames,
  executeTool,
} from './tools.js';
import {
  formatToolResult,
} from './utils.js';

dotenv.config();

// ============ Configuration ============
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const memoryPath = path.join(__dirname, '..', 'memory.md');
const memorySystemPath = path.join(__dirname, '..', 'memory-system.md');

// Constants
const DEFAULT_TIMEOUT_MS = 120_000;
const MAX_RETRIES = 3;
const RETRY_DELAYS = [1000, 2000, 4000];
const MAX_HISTORY_LENGTH = 100;
const MAX_TOKEN_OUTPUT = 4096;
const MAX_TOOL_LOOPS = 10;

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

function recordRequest(inputTokens: number, outputTokens: number, toolCallCount: number): void {
  metrics.requestCount++;
  metrics.totalTokens += inputTokens + outputTokens;
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

interface StreamResult {
  textContent: string;
  toolCalls: ToolCall[];
  usage?: { input_tokens: number; output_tokens: number };
}

// ============ Error Handling ============
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

// ============ Memory ============
async function loadAllMemories(): Promise<string> {
  const memories: string[] = [];

  // 1. Load built-in system memory (immutable instructions)
  try {
    const systemMemory = await readFile(memorySystemPath, 'utf-8');
    memories.push(systemMemory);
  } catch { /* skip if not exists */ }

  // 2. Load user main memory
  try {
    const mainMemory = await readFile(memoryPath, 'utf-8');
    memories.push(mainMemory);
  } catch {
    memories.push('# Agent Memory\n\nNo permanent memory available.');
  }

  return memories.join('\n\n---\n\n');
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

// ============ Streaming ============
async function streamMessages(
  messages: Message[],
  systemPrompt: string,
  debug: boolean = false,
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
          process.stdout.write(`${formatToolResult(argsRaw, 100, debug)}\n`);
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

  recordRequest(inputTokens, outputTokens, parsedToolCalls.length);

  return { textContent, toolCalls: parsedToolCalls, usage: { input_tokens: inputTokens, output_tokens: outputTokens } };
}

// ============ Interactive Mode ============
async function interactiveMode(systemPrompt: string, debug: boolean = false): Promise<void> {
  console.log('\n=== AI Agent Demo ===');
  console.log(`Tools: ${toolNames.join(', ')}`);
  if (debug) console.log('Debug mode: ON');
  console.log('Commands: /new (clear history), /exit (quit)\n');

  const input = getReadline();

  while (true) {
    const userInput = await new Promise<string>(resolve => input.question('\n> ', resolve));
    if (!userInput.trim()) continue;

    // Slash commands
    if (userInput === '/new') { clearHistory(); console.log('[New conversation]\n'); continue; }
    if (userInput === '/exit') { closeReadline(); return; }

    await processUserInput(userInput, systemPrompt, isDebug);
  }
}

// ============ Non-Interactive Mode ============
async function nonInteractiveMode(prompt: string, systemPrompt: string, debug: boolean = false): Promise<void> {
  console.log(`[Single-shot Mode] Executing: "${prompt}"`);
  if (debug) console.log('[Debug mode: ON]\n');
  else console.log();
  await processUserInput(prompt, systemPrompt);
  closeReadline();
}

// ============ Process User Input ============
async function processUserInput(input: string, systemPrompt: string, debug: boolean = false): Promise<void> {
  let loopCount = 0;
  let lastError = '';

  // Build initial messages
  const messages: Message[] = [...history, { role: 'user', content: input }];

  while (loopCount < MAX_TOOL_LOOPS) {
    loopCount++;
    let success = false;

    for (let attempt = 0; attempt <= MAX_RETRIES && !success; attempt++) {
      if (attempt > 0) {
        showStatus(`Retrying (${attempt}/${MAX_RETRIES})`);
        console.log(`\n[Retry ${attempt}] ${lastError}`);
        await sleep(RETRY_DELAYS[attempt - 1] || 4000);
      }

      try {
        showStatus('Sending request...');
        console.log('\n[LLM Response]\n');

        const { textContent, toolCalls } = await streamMessages(messages, systemPrompt, debug);

        // No tools - simple response, exit loop
        if (toolCalls.length === 0) {
          addToHistory({ role: 'user', content: input });
          addToHistory({ role: 'assistant', content: textContent });
          success = true;
          showStatus('Done');
          console.log();
          return;
        }

        // Execute tools sequentially
        console.log();
        const results: string[] = [];
        for (const tc of toolCalls) {
          process.stdout.write(`[Calling ${tc.name}]\n`);
          const result = await executeTool(tc.name, tc.args);
          results.push(result);
          console.log(`[${tc.name} result]\n${formatToolResult(result, 100, debug)}\n`);
        }

        // Build assistant message with tool uses
        const assistantMsg: Message = {
          role: 'assistant',
          content: toolCalls.map((tc) => ({
            type: 'tool_use' as const,
            id: tc.id,
            name: tc.name,
            input: tc.args,
          })),
        };

        // Build tool result messages
        const toolResultMsgs: Message[] = toolCalls.map((tc, i) => ({
          role: 'user' as const,
          content: [{ type: 'tool_result' as const, tool_use_id: tc.id, content: results[i] }],
        }));

        // Update history with this round
        addToHistory({ role: 'user', content: input });
        addToHistory(assistantMsg);
        addToHistory(...toolResultMsgs);

        // Update messages for next iteration
        messages.push({ role: 'user', content: input }, assistantMsg, ...toolResultMsgs);

        // Continue loop to let LLM process tool results
        success = true;
        showStatus('Tool results sent, continuing...');

      } catch (error) {
        recordError();
        const { message, isRetryable } = parseAPIError(error);
        lastError = message;
        if (!isRetryable || attempt >= MAX_RETRIES) {
          console.error(`\n[Error] ${message}\n`);
          return;
        }
      }
    }
  }

  if (loopCount >= MAX_TOOL_LOOPS) {
    console.log('[Warning] Reached max tool loop limit');
  }
}

// ============ Main ============
async function main(): Promise<void> {
  // Parse command line arguments
  const { values, positionals } = parseArgs({
    args: process.argv.slice(2),
    options: {
      help: { type: 'boolean', default: false, short: 'h' },
      debug: { type: 'boolean', default: false, short: 'd' },
    },
    allowPositionals: true,
  });

  const isDebug = values.debug === true;

  if (values.help) {
    console.log(`
Usage: agent-demo [OPTIONS] [PROMPT]

Options:
  -h, --help     Show this help message
  -d, --debug    Show full output without truncation (thinking blocks, tool results)

Arguments:
  PROMPT         User prompt to execute in single-shot mode (non-interactive)

If PROMPT is provided, runs in non-interactive mode and exits after execution.
Otherwise, runs in interactive REPL mode.

Examples:
  agent-demo "List files in current directory"
  agent-demo --debug "Trace the execution"
  agent-demo --help
`);
    closeReadline();
    return;
  }

  const systemPrompt = await loadAllMemories();
  const prompt = positionals.join(' ');

  if (prompt) {
    await nonInteractiveMode(prompt, systemPrompt, isDebug);
  } else {
    await interactiveMode(systemPrompt, isDebug);
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
