import Anthropic from '@anthropic-ai/sdk';
import dotenv from 'dotenv';

dotenv.config();

const baseURL = process.env.ANTHROPIC_BASE_URL || 'https://api.minimaxi.com/anthropic';
const apiKey = process.env.ANTHROPIC_API_KEY || '';
const model = process.env.ANTHROPIC_MODEL || 'MiniMax-M2.7';

if (!apiKey) {
  console.error('Error: ANTHROPIC_API_KEY is not set in .env file');
  process.exit(1);
}

console.log(`[Config] API: ${baseURL}`);
console.log(`[Config] Model: ${model}`);

export const anthropic = new Anthropic({
  baseURL,
  apiKey,
});

export { model };
