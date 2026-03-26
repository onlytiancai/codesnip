import dotenv from 'dotenv';
import { z } from 'zod';

dotenv.config();

const configSchema = z.object({
  // LLM Configuration
  LLM_BASE_URL: z.string().default('https://api.minimaxi.com/anthropic'),
  LLM_API_KEY: z.string(),
  LLM_MODEL: z.string().default('MiniMax-M2.7'),

  // TTS Configuration
  TTS_PROVIDER: z.enum(['edge', 'bytedance']).default('edge'),

  // ByteDance TTS
  BYTEDANCE_APP_ID: z.string().optional(),
  BYTEDANCE_ACCESS_KEY: z.string().optional(),
  BYTEDANCE_RESOURCE_ID: z.string().optional(),

  // Server
  PORT: z.string().default('3000'),
});

const parsed = configSchema.safeParse(process.env);

if (!parsed.success) {
  console.error('Invalid configuration:', parsed.error.flatten().fieldErrors);
  process.exit(1);
}

export const config = parsed.data;

export type TTSProvider = 'edge' | 'bytedance';
