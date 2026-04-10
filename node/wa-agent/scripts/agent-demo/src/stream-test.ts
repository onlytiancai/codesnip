import Anthropic from '@anthropic-ai/sdk';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load config from scripts/.env (which is a symlink to ../../.env)
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
        content: 'Count to 5, one number per line.',
      },
    ],
  });

  let fullContent = '';

  for await (const event of stream) {
    if (event.type === 'message_start') {
      console.log('Message started\n');
    } else if (event.type === 'content_block_start') {
      console.log('Content block started');
    } else if (event.type === 'content_block_delta') {
      const text = event.delta.type === 'text_delta' ? event.delta.text : '';
      fullContent += text;
      process.stdout.write(text);
    } else if (event.type === 'content_block_stop') {
      console.log('\nContent block stopped');
    } else if (event.type === 'message_delta') {
      console.log(`\nMessage finished. Stop reason: ${event.delta.stop_reason}`);
    } else if (event.type === 'message_stop') {
      console.log('Stream complete');
    }
  }

  console.log('\n--- Full content ---');
  console.log(fullContent);
}

main().catch(console.error);
