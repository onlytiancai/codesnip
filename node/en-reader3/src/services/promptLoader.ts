import { readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../utils/logger.js';
import type { PromptsConfig } from '../types/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let cachedConfig: PromptsConfig | null = null;

/**
 * Load and validate prompts configuration from config/prompts.json
 */
export async function loadPromptsConfig(): Promise<PromptsConfig> {
  if (cachedConfig) {
    return cachedConfig;
  }

  const configPath = join(__dirname, '..', '..', 'config', 'prompts.json');

  try {
    const content = await readFile(configPath, 'utf-8');
    const config = JSON.parse(content) as PromptsConfig;

    validateConfig(config);

    cachedConfig = config;
    logger.debug('Prompts config loaded and validated successfully');

    return config;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      logger.error(`Prompts config not found at ${configPath}`);
    } else if (error instanceof SyntaxError) {
      logger.error('Invalid JSON in prompts config');
    } else {
      logger.error('Failed to load prompts config:', error);
    }
    throw error;
  }
}

/**
 * Validate prompts configuration structure
 */
function validateConfig(config: PromptsConfig): void {
  const errors: string[] = [];

  if (!config.systemPrompt || typeof config.systemPrompt !== 'string') {
    errors.push('Missing or invalid systemPrompt');
  }

  if (typeof config.temperature !== 'number' || config.temperature < 0 || config.temperature > 2) {
    errors.push('Missing or invalid temperature (should be 0-2)');
  }

  if (!config.prompts || typeof config.prompts !== 'object') {
    errors.push('Missing or invalid prompts object');
    throw new Error(`Prompts config validation failed: ${errors.join(', ')}`);
  }

  const requiredPrompts = ['intro', 'outro', 'segment'];
  for (const promptName of requiredPrompts) {
    if (!config.prompts[promptName as keyof typeof config.prompts]) {
      errors.push(`Missing prompt: ${promptName}`);
    } else {
      const prompt = config.prompts[promptName as keyof typeof config.prompts];
      if (!prompt.template || typeof prompt.template !== 'string') {
        errors.push(`Missing or invalid template for ${promptName}`);
      }
      if (!prompt.responseFormat || typeof prompt.responseFormat !== 'string') {
        errors.push(`Missing or invalid responseFormat for ${promptName}`);
      }
    }
  }

  if (errors.length > 0) {
    throw new Error(`Prompts config validation failed: ${errors.join(', ')}`);
  }
}

/**
 * Interpolate template with variables
 * Replaces {{variableName}} placeholders with provided values
 */
export function interpolateTemplate(template: string, variables: Record<string, string | number>): string {
  let result = template;

  for (const [key, value] of Object.entries(variables)) {
    const placeholder = `{{${key}}}`;
    result = result.split(placeholder).join(String(value));
  }

  return result;
}

/**
 * Clear cached config (useful for testing or config reload)
 */
export function clearConfigCache(): void {
  cachedConfig = null;
}
