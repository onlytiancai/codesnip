import Anthropic from '@anthropic-ai/sdk';
import { readFile, writeFile, mkdir, readdir, stat } from 'fs/promises';
import * as readline from 'readline';
import * as cheerio from 'cheerio';
import { z } from 'zod';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

// ============ Configuration ============
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const memoryPath = path.join(__dirname, '..', 'memory.md');

const baseURL = process.env.ANTHROPIC_BASE_URL || 'https://api.anthropic.com';
const apiKey = process.env.ANTHROPIC_API_KEY || '';
const model = process.env.ANTHROPIC_MODEL || 'claude-3-5-sonnet-20241022';

if (!apiKey) {
  console.error('Error: ANTHROPIC_API_KEY is not set');
  process.exit(1);
}

console.log(`[Config] API: ${baseURL}`);
console.log(`[Config] Model: ${model}`);

const anthropic = new Anthropic({ baseURL, apiKey });

// ============ Types ============
interface Message {
  role: 'user' | 'assistant';
  content: string | ContentBlock[];
}

type ContentBlock =
  | { type: 'tool_use'; id: string; name: string; input: Record<string, any> }
  | { type: 'tool_result'; tool_use_id: string; content: string };

interface ToolCall {
  name: string;
  id: string;
  args: Record<string, any>;
}

interface StreamResult {
  textContent: string;
  toolCalls: ToolCall[];
}

// ============ Tools ============
const tools = {
  read: {
    description: 'Read the contents of a file',
    schema: z.object({ path: z.string().describe('The file path to read') }),
    async execute({ path: filePath }: { path: string }) {
      try {
        return await readFile(filePath, 'utf-8');
      } catch (error) {
        return `Error: ${error instanceof Error ? error.message : String(error)}`;
      }
    },
  },
  write: {
    description: 'Write content to a file',
    schema: z.object({
      path: z.string().describe('The file path to write'),
      content: z.string().describe('The content to write'),
    }),
    async execute({ path: filePath, content }: { path: string; content: string }) {
      try {
        await mkdir(path.dirname(filePath), { recursive: true });
        await writeFile(filePath, content, 'utf-8');
        return `Successfully wrote to ${filePath}`;
      } catch (error) {
        return `Error: ${error instanceof Error ? error.message : String(error)}`;
      }
    },
  },
  ls: {
    description: 'List directory contents',
    schema: z.object({ path: z.string().describe('The directory path to list') }),
    async execute({ path: dirPath }: { path: string }) {
      try {
        const entries = await readdir(dirPath);
        const result = await Promise.all(
          entries.map(async (name) => {
            const fullPath = path.join(dirPath, name);
            const s = await stat(fullPath);
            return `${s.isDirectory() ? 'd' : '-'} ${name}`;
          })
        );
        return result.join('\n');
      } catch (error) {
        return `Error: ${error instanceof Error ? error.message : String(error)}`;
      }
    },
  },
  mkdir: {
    description: 'Create a directory',
    schema: z.object({ path: z.string().describe('The directory path to create') }),
    async execute({ path: dirPath }: { path: string }) {
      try {
        await mkdir(dirPath, { recursive: true });
        return `Successfully created directory: ${dirPath}`;
      } catch (error) {
        return `Error: ${error instanceof Error ? error.message : String(error)}`;
      }
    },
  },
  web_fetch: {
    description: 'Fetch content from a web URL',
    schema: z.object({ url: z.string().url().describe('The URL to fetch') }),
    async execute({ url }: { url: string }) {
      try {
        const response = await fetch(url, {
          headers: { 'User-Agent': 'AI-Agent/1.0' },
          signal: AbortSignal.timeout(30000),
        });
        const html = await response.text();
        return cheerio.load(html)('body').text().trim().slice(0, 5000);
      } catch (error) {
        return `Error: ${error instanceof Error ? error.message : String(error)}`;
      }
    },
  },
};

const toolDefinitions: any[] = Object.entries(tools).map(([name, tool]) => ({
  name,
  description: tool.description,
  input_schema: (tool as any).schema.shape,
}));

async function executeTool(name: string, args: any): Promise<string> {
  const tool = tools[name as keyof typeof tools];
  if (!tool) return `Unknown tool: ${name}`;
  return (tool as any).execute(args);
}

// ============ Memory ============
async function loadMemory(): Promise<string> {
  try {
    return await readFile(memoryPath, 'utf-8');
  } catch {
    return '# Agent Memory\n\nNo persistent memory available.';
  }
}

// ============ Helpers ============
const history: Message[] = [];

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const clearLine = () => process.stdout.write('\r\x1b[K');

const showStatus = (status: string) => {
  clearLine();
  process.stdout.write(`[${new Date().toLocaleTimeString()}] ${status}`);
};

const parseAPIError = (error: unknown): { message: string; isRetryable: boolean } => {
  if (error instanceof Error) {
    const msg = error.message;
    if (msg.includes('429') || msg.includes('rate limit')) return { message: 'Rate limit, retrying...', isRetryable: true };
    if (msg.includes('500') || msg.includes('502') || msg.includes('503')) return { message: 'Server error, retrying...', isRetryable: true };
    if (msg.includes('network') || msg.includes('ECONNREFUSED')) return { message: 'Network error, retrying...', isRetryable: true };
    return { message: msg, isRetryable: false };
  }
  return { message: String(error), isRetryable: false };
};

const MAX_RETRIES = 3;
const RETRY_DELAYS = [1000, 2000, 4000];

// ============ Streaming ============
async function streamMessages(
  messages: Message[],
  systemPrompt: string,
): Promise<StreamResult> {
  const toolCalls: { name: string; id: string; argsRaw: string }[] = [];
  let currentToolIndex = -1;
  let isThinking = false;
  let textContent = '';

  const stream = anthropic.messages.stream({
    model,
    max_tokens: 4096,
    system: systemPrompt,
    messages: messages as any,
    tools: toolDefinitions as any,
  });

  for await (const event of stream) {
    const e = event as any;

    if (e.type === 'message_stop') {
      console.log();
    } else if (e.type === 'content_block_start') {
      if (e.content_block?.type === 'thinking') {
        isThinking = true;
        process.stdout.write('\n[Thinking]\n');
      } else if (e.content_block?.type === 'tool_use') {
        currentToolIndex++;
        toolCalls.push({
          name: e.content_block.name || '',
          id: e.content_block.id || '',
          argsRaw: '',
        });
      } else if (e.content_block?.type === 'text') {
        if (isThinking) { console.log('\n[/Thinking]'); isThinking = false; }
        process.stdout.write('\n[Answer]\n');
      }
    } else if (e.type === 'content_block_delta') {
      if (e.delta?.type === 'thinking_delta') {
        if (!isThinking) { isThinking = true; process.stdout.write('\n[Thinking]\n'); }
        process.stdout.write(e.delta.thinking || '');
      } else if (e.delta?.type === 'input_json_delta') {
        if (currentToolIndex >= 0 && toolCalls[currentToolIndex]) {
          toolCalls[currentToolIndex].argsRaw += e.delta.partial_json || '';
        }
      } else if (e.delta?.type === 'text_delta' && e.delta?.text) {
        if (isThinking) { console.log('\n[/Thinking]'); isThinking = false; }
        process.stdout.write(e.delta.text);
        textContent += e.delta.text;
      }
    } else if (e.type === 'message_delta' && e.delta?.text) {
      if (isThinking) { console.log('\n[/Thinking]'); isThinking = false; }
      process.stdout.write(e.delta.text);
      textContent += e.delta.text;
    }
  }

  // Parse tool args after stream completes
  const parsedToolCalls: ToolCall[] = toolCalls.map(tc => ({
    name: tc.name,
    id: tc.id,
    args: JSON.parse(tc.argsRaw),
  }));

  return { textContent, toolCalls: parsedToolCalls };
}

// ============ Main ============
async function main() {
  const systemPrompt = await loadMemory();

  console.log('\n=== AI Agent Demo ===');
  console.log('Tools: read, write, ls, mkdir, web_fetch');
  console.log('Commands: /new (clear history), /exit (quit)\n');

  while (true) {
    const input = await new Promise<string>(resolve => rl.question('\n> ', resolve));
    if (!input.trim()) continue;

    // Slash commands
    if (input === '/new') { history.length = 0; console.log('[New conversation]\n'); continue; }
    if (input === '/exit') process.exit(0);

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

        // First request
        const { textContent, toolCalls } = await streamMessages(
          [...history, { role: 'user', content: input }],
          systemPrompt
        );

        // No tools - simple response
        if (toolCalls.length === 0) {
          history.push({ role: 'user', content: input });
          history.push({ role: 'assistant', content: textContent });
          success = true;
          showStatus('Done');
          console.log();
          continue;
        }

        // Execute tools in parallel
        console.log();
        toolCalls.forEach(tc => process.stdout.write(`[Calling ${tc.name}]\n`));

        const results = await Promise.all(toolCalls.map(tc => executeTool(tc.name, tc.args)));
        results.forEach((r, i) => console.log(`[${toolCalls[i].name} result]\n${r}\n`));

        // Build messages for follow-up
        const assistantMsg: Message = {
          role: 'assistant',
          content: toolCalls.map((tc, i) => ({
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
        history.push({ role: 'user', content: input });
        history.push(assistantMsg);
        history.push(...toolResultMsgs);

        // Handle follow-up tools (simplified single level)
        if (followUpTools.length > 0) {
          const followUpResults = await Promise.all(followUpTools.map(tc => executeTool(tc.name, tc.args)));
          history.push({
            role: 'assistant',
            content: followUpTools.map((tc, i) => ({
              type: 'tool_use' as const,
              id: tc.id,
              name: tc.name,
              input: tc.args,
            })),
          });
          followUpTools.forEach((tc, i) => {
            history.push({
              role: 'user' as const,
              content: [{ type: 'tool_result' as const, tool_use_id: tc.id, content: followUpResults[i] }],
            });
          });
        }

        history.push({ role: 'assistant', content: finalText || textContent });
        success = true;
        showStatus('Done');
        console.log();

      } catch (error) {
        const { message, isRetryable } = parseAPIError(error);
        lastError = message;
        if (!isRetryable || attempt >= MAX_RETRIES) {
          console.error(`\n[Error] ${message}\n`);
          break;
        }
      }
    }
  }
}

process.on('SIGINT', () => { console.log('\n\nExiting...'); process.exit(0); });

main().catch(console.error);
