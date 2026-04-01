import { readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../utils/logger.js';
import type { PromptsConfig, PromptDefinition } from '../types/index.js';

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
    await loadExternalTemplates(config);

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
 * Load external template files referenced by templateFile field
 */
async function loadExternalTemplates(config: PromptsConfig): Promise<void> {
  const baseDir = join(__dirname, '..', '..', 'config');

  for (const [promptName, promptDef] of Object.entries(config.prompts)) {
    if (promptDef.templateFile) {
      const templatePath = join(baseDir, promptDef.templateFile);
      try {
        const templateContent = await readFile(templatePath, 'utf-8');
        (promptDef as PromptDefinition).template = templateContent;
        logger.debug(`Loaded external template for ${promptName} from ${promptDef.templateFile}`);
      } catch (error) {
        logger.error(`Failed to load template file for ${promptName}: ${error}`);
        throw error;
      }
    }
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
    const prompt = config.prompts[promptName as keyof typeof config.prompts];
    if (!prompt) {
      errors.push(`Missing prompt: ${promptName}`);
    } else {
      const hasTemplate = prompt.template && typeof prompt.template === 'string';
      const hasTemplateFile = prompt.templateFile && typeof prompt.templateFile === 'string';
      if (!hasTemplate && !hasTemplateFile) {
        errors.push(`Missing template or templateFile for ${promptName}`);
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
