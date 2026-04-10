import { readFile, writeFile } from 'fs/promises';
import * as readline from 'readline';
import { anthropic, model } from './client.js';
import { tools } from './tools.js';
import dotenv from 'dotenv';

dotenv.config();

const memoryPath = './memory.md';

async function loadMemory(): Promise<string> {
  try {
    const content = await readFile(memoryPath, 'utf-8');
    return content;
  } catch {
    return '# Agent Memory\n\nNo persistent memory available.';
  }
}

async function saveMemory(content: string): Promise<void> {
  await writeFile(memoryPath, content, 'utf-8');
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const history: Message[] = [];

function createReadlineInterface() {
  return readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
}

function clearLine() {
  process.stdout.write('\r\x1b[K');
}

function showStatus(status: string) {
  clearLine();
  process.stdout.write(`[${new Date().toLocaleTimeString()}] ${status}`);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function parseAPIError(error: unknown): { message: string; isRetryable: boolean } {
  if (error instanceof Error) {
    const msg = error.message;
    // Check for common retryable errors
    if (msg.includes('429') || msg.includes('rate limit') || msg.includes('timeout')) {
      return { message: 'API rate limit exceeded, will retry...', isRetryable: true };
    }
    if (msg.includes('500') || msg.includes('502') || msg.includes('503')) {
      return { message: 'Server error, will retry...', isRetryable: true };
    }
    if (msg.includes('network') || msg.includes('ECONNREFUSED')) {
      return { message: 'Network error, will retry...', isRetryable: true };
    }
    return { message: msg, isRetryable: false };
  }
  return { message: String(error), isRetryable: false };
}

const MAX_RETRIES = 3;
const RETRY_DELAYS = [1000, 2000, 4000]; // ms

async function main() {
  const memory = await loadMemory();
  const systemPrompt = memory;

  console.log('\n=== AI Agent Demo ===');
  console.log('Type your messages and press Enter to chat.');
  console.log('Available tools: read, write, ls, mkdir, web_fetch');
  console.log('Press Ctrl+C to exit.\n');

  const rl = createReadlineInterface();

  while (true) {
    const input = await new Promise<string>((resolve) => {
      rl.question('\n> ', resolve);
    });

    if (!input.trim()) continue;

    let fullResponse = '';
    let requestSuccess = false;
    let lastError = '';

    for (let attempt = 0; attempt <= MAX_RETRIES && !requestSuccess; attempt++) {
      if (attempt > 0) {
        const delay = RETRY_DELAYS[attempt - 1] || RETRY_DELAYS[RETRY_DELAYS.length - 1];
        showStatus(`Retrying... (${attempt}/${MAX_RETRIES})`);
        console.log(`\n[Retry ${attempt}/${MAX_RETRIES}] ${lastError}`);
        await sleep(delay);
      }

      try {
        showStatus('Sending request to LLM...');
        console.log();

        const stream = anthropic.messages.stream({
          model: model,
          max_tokens: 4096,
          system: systemPrompt,
          messages: [
            ...history,
            { role: 'user', content: input },
          ],
          tools: [
            {
              name: 'read',
              description: tools.read.description,
              input_schema: {
                type: 'object',
                properties: {
                  path: { type: 'string', description: 'The file path to read' }
                },
                required: ['path']
              },
            },
            {
              name: 'write',
              description: tools.write.description,
              input_schema: {
                type: 'object',
                properties: {
                  path: { type: 'string', description: 'The file path to write' },
                  content: { type: 'string', description: 'The content to write' }
                },
                required: ['path', 'content']
              },
            },
            {
              name: 'ls',
              description: tools.ls.description,
              input_schema: {
                type: 'object',
                properties: {
                  path: { type: 'string', description: 'The directory path to list' }
                },
                required: ['path']
              },
            },
            {
              name: 'mkdir',
              description: tools.mkdir.description,
              input_schema: {
                type: 'object',
                properties: {
                  path: { type: 'string', description: 'The directory path to create' }
                },
                required: ['path']
              },
            },
            {
              name: 'web_fetch',
              description: tools.web_fetch.description,
              input_schema: {
                type: 'object',
                properties: {
                  url: { type: 'string', description: 'The URL to fetch', format: 'uri' }
                },
                required: ['url']
              },
            },
          ],
        });

        clearLine();
        console.log('[LLM Response]');
        console.log();

        let isThinking = false;
        let thinkingContent = '';
        let currentTextContent = '';
        let toolName = '';
        let toolId = '';
        let toolArgs = '';
        let toolWasCalled = false;

        for await (const event of stream) {
          const e = event as any;

          if (e.type === 'message_stop') {
            console.log();
          } else if (e.type === 'message_delta') {
            if (e.delta?.text) {
              process.stdout.write(e.delta.text);
              fullResponse += e.delta.text;
            }
          } else if (e.type === 'content_block_start') {
            if (e.content_block?.type === 'thinking') {
              isThinking = true;
              thinkingContent = e.content_block?.thinking || '';
              process.stdout.write('\n[Thinking]\n');
            } else if (e.content_block?.type === 'text') {
              if (isThinking) {
                console.log('\n[/Thinking]');
              }
              isThinking = false;
              currentTextContent = '';
              process.stdout.write('\n[Answer]\n');
            } else if (e.content_block?.type === 'tool_use') {
              toolName = e.content_block.name || '';
              toolId = e.content_block.id || '';
              toolArgs = '';
            }
          } else if (e.type === 'content_block_delta') {
            if (e.delta?.type === 'text_delta' && e.delta?.text) {
              process.stdout.write(e.delta.text);
              fullResponse += e.delta.text;
              currentTextContent += e.delta.text;
            } else if (e.delta?.type === 'thinking_delta') {
              if (!isThinking) {
                isThinking = true;
                process.stdout.write('\n[Thinking]\n');
              }
              if (e.delta?.thinking) {
                process.stdout.write(e.delta.thinking);
                thinkingContent += e.delta.thinking;
              }
            } else if (e.delta?.type === 'input_json_delta') {
              toolArgs += e.delta.partial_json || '';
            }
          } else if (e.type === 'content_block_stop') {
            // Tool call completed, execute it
            if (toolName && toolId) {
              console.log('\n');
              process.stdout.write(`[Calling tool: ${toolName}]\n`);
              let toolResult = '';
              try {
                const args = JSON.parse(toolArgs);
                if (toolName === 'read') {
                  toolResult = await tools.read.execute(args);
                } else if (toolName === 'write') {
                  toolResult = await tools.write.execute(args);
                } else if (toolName === 'ls') {
                  toolResult = await tools.ls.execute(args);
                } else if (toolName === 'mkdir') {
                  toolResult = await tools.mkdir.execute(args);
                } else if (toolName === 'web_fetch') {
                  toolResult = await tools.web_fetch.execute(args);
                } else {
                  toolResult = `Unknown tool: ${toolName}`;
                }
              } catch (parseError) {
                toolResult = `Error parsing tool arguments: ${parseError instanceof Error ? parseError.message : String(parseError)}`;
              }
              console.log(`[Tool result]\n${toolResult}\n`);

              // Send tool result back to model
              showStatus('Sending tool result to LLM...');
              const toolMessage = {
                role: 'user' as const,
                content: [{
                  type: 'tool_result',
                  tool_use_id: toolId,
                  content: toolResult,
                }],
              };

              // Build assistant message with tool_use to include in history
              const assistantToolMessage = {
                role: 'assistant' as const,
                content: [{
                  type: 'tool_use' as const,
                  id: toolId,
                  name: toolName,
                  input: JSON.parse(toolArgs),
                }],
              };

              const followUpStream = anthropic.messages.stream({
                model: model,
                max_tokens: 4096,
                system: systemPrompt,
                messages: [
                  { role: 'user', content: input },
                  assistantToolMessage,
                  toolMessage,
                ],
                tools: [
                  { name: 'read', description: tools.read.description, input_schema: { type: 'object', properties: { path: { type: 'string', description: 'The file path to read' } }, required: ['path'] } },
                  { name: 'write', description: tools.write.description, input_schema: { type: 'object', properties: { path: { type: 'string', description: 'The file path to write' }, content: { type: 'string', description: 'The content to write' } }, required: ['path', 'content'] } },
                  { name: 'ls', description: tools.ls.description, input_schema: { type: 'object', properties: { path: { type: 'string', description: 'The directory path to list' } }, required: ['path'] } },
                  { name: 'mkdir', description: tools.mkdir.description, input_schema: { type: 'object', properties: { path: { type: 'string', description: 'The directory path to create' } }, required: ['path'] } },
                  { name: 'web_fetch', description: tools.web_fetch.description, input_schema: { type: 'object', properties: { url: { type: 'string', description: 'The URL to fetch', format: 'uri' } }, required: ['url'] } },
                ],
              });

              clearLine();
              console.log('[LLM Response]');
              console.log();

              let followUpResponse = '';
              for await (const followUpEvent of followUpStream) {
                const fe = followUpEvent as any;
                if (fe.type === 'message_delta') {
                  if (fe.delta?.text) {
                    process.stdout.write(fe.delta.text);
                    followUpResponse += fe.delta.text;
                  }
                } else if (fe.type === 'content_block_delta') {
                  if (fe.delta?.type === 'text_delta' && fe.delta?.text) {
                    process.stdout.write(fe.delta.text);
                    followUpResponse += fe.delta.text;
                  }
                } else if (fe.type === 'message_stop') {
                  console.log();
                }
              }

              fullResponse = followUpResponse;
              toolWasCalled = true;
              toolName = '';
              toolId = '';
              toolArgs = '';

              // Add all messages to history after tool call completes
              history.push({ role: 'user', content: input });
              history.push({ role: 'assistant', content: fullResponse });
            }
          }
        }

        // Add to history if no tool was called
        if (!toolWasCalled) {
          history.push({ role: 'user', content: input });
          history.push({ role: 'assistant', content: fullResponse });
        }

        requestSuccess = true;

        console.log();
        clearLine();
        showStatus('Response complete');
        console.log();
        console.log();
      } catch (error) {
        const { message, isRetryable } = parseAPIError(error);
        lastError = message;

        if (!isRetryable || attempt >= MAX_RETRIES) {
          clearLine();
          console.error(`\n[Error] ${message}`);
          if (isRetryable && attempt >= MAX_RETRIES) {
            console.error('[Error] Max retries reached, giving up.');
          }
          // Still add to history if we got partial response
          if (fullResponse) {
            history.push({ role: 'user', content: input });
            history.push({ role: 'assistant', content: fullResponse });
          }
          console.log();
          break;
        }
        // If retryable, the for loop will continue
      }
    }
  }
}

process.on('SIGINT', async () => {
  console.log('\n\nExiting...');
  console.log(`\nYou had ${Math.floor(history.length / 2)} conversation turns.`);

  const rl = createReadlineInterface();
  const answer = await new Promise<string>((resolve) => {
    rl.question('Would you like to update memory.md? (y/n): ', resolve);
  });

  if (answer.toLowerCase() === 'y') {
    const newMemory = await new Promise<string>((resolve) => {
      rl.question('Enter new memory content (or press Enter to keep current): ', resolve);
    });

    if (newMemory.trim()) {
      await saveMemory(newMemory);
      console.log('Memory updated!');
    }
  }

  process.exit(0);
});

main().catch(console.error);
