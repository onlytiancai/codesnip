import Anthropic from '@anthropic-ai/sdk';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load config from scripts/.env
dotenv.config({ path: path.join(__dirname, '..', '.env') });

const apiKey = process.env.ANTHROPIC_API_KEY;
const baseURL = process.env.ANTHROPIC_BASE_URL;
const model = process.env.ANTHROPIC_MODEL || 'MiniMax-M2';

if (!apiKey) {
  console.error('ANTHROPIC_API_KEY is not set');
  process.exit(1);
}

const client = new Anthropic({
  apiKey,
  baseURL: baseURL || undefined,
});

async function main() {
  console.log('--- Streaming Test ---\n');

  const stream = await client.messages.stream({
    model,
    max_tokens: 1024,
    messages: [
      {
        role: 'user',
        content: '你好',
      },
    ],
  });

  for await (const event of stream) {
    if (event.type === 'content_block_start') {
      if (event.content_block.type === 'thinking') {
        process.stdout.write('[Thinking]\n');
      } else if (event.content_block.type === 'text') {
        process.stdout.write('[Text]\n');
      }
    } else if (event.type === 'content_block_delta') {
      if (event.delta.type === 'thinking_delta') {
        process.stdout.write(event.delta.thinking);
      } else if (event.delta.type === 'text_delta') {
        process.stdout.write(event.delta.text);
      }
    } else if (event.type === 'content_block_stop') {
      process.stdout.write('\n');
      if (event.index === 0) {
        process.stdout.write('[/Thinking]\n');
      } else {
        process.stdout.write('[/Text]\n');
      }
    }
  }

  console.log();
}

main().catch(console.error);
