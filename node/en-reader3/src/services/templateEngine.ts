import { readFile, readdir } from 'fs/promises';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { logger } from '../utils/logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const TEMPLATES_DIR = join(__dirname, '..', '..', 'templates');

/**
 * Render a template with placeholder replacements.
 * Placeholders use {{placeholder}} syntax.
 */
export async function renderTemplate(
  templateName: string,
  data: Record<string, string | string[] | number | boolean | undefined>
): Promise<string> {
  const templatePath = join(TEMPLATES_DIR, templateName);

  try {
    const template = await readFile(templatePath, 'utf-8');
    return replacePlaceholders(template, data);
  } catch (err) {
    logger.error(`Failed to read template ${templateName}:`, err);
    throw new Error(`Template not found: ${templateName}`);
  }
}

/**
 * Replace {{placeholder}} with values from data object.
 */
function replacePlaceholders(
  template: string,
  data: Record<string, string | string[] | number | boolean | undefined>
): string {
  let result = template;

  for (const [key, value] of Object.entries(data)) {
    const placeholder = `{{${key}}}`;
    const replacement = value !== undefined ? String(value) : '';
    result = result.split(placeholder).join(replacement);
  }

  // Remove any remaining placeholders (optional: could throw error instead)
  result = result.replace(/\{\{[^}]+\}\}/g, '');

  return result;
}

/**
 * List available templates.
 */
export async function listTemplates(): Promise<string[]> {
  try {
    const files = await readdir(TEMPLATES_DIR);
    return files.filter((f) => f.endsWith('.html'));
  } catch {
    return [];
  }
}
