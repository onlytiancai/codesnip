import { readFile, writeFile, mkdir, readdir, stat } from 'fs/promises';
import * as cheerio from 'cheerio';
import { z } from 'zod';
import path from 'path';
import { fileURLToPath } from 'url';

// ============ Constants ============
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const WEB_FETCH_MAX_CHARS = 5000;
const ALLOWED_BASE_DIR = process.cwd();
const memoryDir = path.join(__dirname, '..', 'memory');

// ============ Types ============
export interface ToolCall {
  name: string;
  id: string;
  args: Record<string, unknown>;
}

export interface ToolDefinition {
  name: string;
  description: string;
  input_schema: z.ZodRawShape;
}

// ============ Path Validation ============
export function validatePath(filePath: string): string {
  const resolved = path.resolve(filePath);
  if (!resolved.startsWith(ALLOWED_BASE_DIR)) {
    throw new Error(`Path ${filePath} is outside allowed directory`);
  }
  return resolved;
}

// ============ Tools ============
const tools: Record<string, {
  description: string;
  schema: z.ZodRawShape;
  execute: (args: Record<string, unknown>) => Promise<string>;
}> = {
  read: {
    description: 'Read the contents of a file',
    schema: { path: z.string().describe('The file path to read') },
    async execute({ path: filePath }) {
      try {
        validatePath(String(filePath));
        return await readFile(String(filePath), 'utf-8');
      } catch (error) {
        throw error;
      }
    },
  },
  write: {
    description: 'Write content to a file',
    schema: {
      path: z.string().describe('The file path to write'),
      content: z.string().describe('The content to write'),
    },
    async execute({ path: filePath, content }) {
      try {
        const resolved = validatePath(String(filePath));
        await mkdir(path.dirname(resolved), { recursive: true });
        await writeFile(resolved, String(content), 'utf-8');
        return `Successfully wrote to ${resolved}`;
      } catch (error) {
        throw error;
      }
    },
  },
  ls: {
    description: 'List directory contents',
    schema: { path: z.string().describe('The directory path to list') },
    async execute({ path: dirPath }) {
      try {
        const resolved = validatePath(String(dirPath));
        const entries = await readdir(resolved);
        const result = await Promise.all(
          entries.map(async (name) => {
            const fullPath = path.join(resolved, name);
            const s = await stat(fullPath);
            return `${s.isDirectory() ? 'd' : '-'} ${name}`;
          })
        );
        return result.join('\n');
      } catch (error) {
        throw error;
      }
    },
  },
  mkdir: {
    description: 'Create a directory',
    schema: { path: z.string().describe('The directory path to create') },
    async execute({ path: dirPath }) {
      try {
        const resolved = validatePath(String(dirPath));
        await mkdir(resolved, { recursive: true });
        return `Successfully created directory: ${resolved}`;
      } catch (error) {
        throw error;
      }
    },
  },
  web_fetch: {
    description: 'Fetch content from a web URL',
    schema: { url: z.string().url().describe('The URL to fetch') },
    async execute({ url }) {
      try {
        const response = await fetch(String(url), {
          headers: { 'User-Agent': 'AI-Agent/1.0' },
          signal: AbortSignal.timeout(30000),
        });
        const html = await response.text();
        const text = cheerio.load(html)('body').text().trim();
        if (text.length > WEB_FETCH_MAX_CHARS) {
          return text.substring(0, WEB_FETCH_MAX_CHARS) + `... (${text.length} chars total)`;
        }
        return text;
      } catch (error) {
        throw error;
      }
    },
  },
  save_today_memory: {
    description: 'Save important information learned today to daily memory file',
    schema: { content: z.string().describe('Content to save') },
    async execute({ content }) {
      const today = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
      const filepath = path.join(memoryDir, `${today}.md`);
      try {
        await mkdir(memoryDir, { recursive: true });
        const existing = await readFile(filepath, 'utf-8').catch(() => '');
        const timestamp = new Date().toISOString();
        const entry = `\n\n## [${timestamp}] Memory Entry\n\n${content}`;
        await writeFile(filepath, existing + entry, 'utf-8');
        return `Saved to today's memory: ${today}.md`;
      } catch (error) {
        throw error;
      }
    },
  },
  save_global_memory: {
    description: 'Save permanent information to global memory (identity, user preferences)',
    schema: { content: z.string().describe('Content to save') },
    async execute({ content }) {
      const filepath = path.join(__dirname, '..', 'memory.md');
      try {
        const existing = await readFile(filepath, 'utf-8').catch(() => '');
        const timestamp = new Date().toISOString();
        const entry = `\n\n## [${timestamp}] Memory Entry\n\n${content}`;
        await writeFile(filepath, existing + entry, 'utf-8');
        return 'Saved to global memory: memory.md';
      } catch (error) {
        throw error;
      }
    },
  },
};

export const toolDefinitions: ToolDefinition[] = Object.entries(tools).map(([name, tool]) => ({
  name,
  description: tool.description,
  input_schema: tool.schema,
}));

export const toolNames = Object.keys(tools);

// ============ Tool Execution ============
export async function executeTool(name: string, args: Record<string, unknown>): Promise<string> {
  const tool = tools[name];
  if (!tool) return `Unknown tool: ${name}`;
  try {
    return await tool.execute(args);
  } catch (error) {
    if (error instanceof Error) {
      return `Error: ${error.message}`;
    }
    return `Error: ${String(error)}`;
  }
}
