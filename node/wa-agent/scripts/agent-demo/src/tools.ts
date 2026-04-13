import { readFile, writeFile, mkdir, readdir, stat, copyFile } from 'fs/promises';
import * as cheerio from 'cheerio';
import { z } from 'zod';
import path from 'path';
import { fileURLToPath } from 'url';

// ============ Constants ============
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const WEB_FETCH_MAX_CHARS = 5000;
const ALLOWED_BASE_DIR = process.cwd();

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
  read_memory: {
    description: 'Read the current memory content',
    schema: {},
    async execute() {
      const memoryPath = path.join(__dirname, '..', 'memory.md');
      const content = await readFile(memoryPath, 'utf-8').catch(() => '');
      return content || '(empty)';
    },
  },
  save_memory: {
    description: 'Save content to memory.md. Automatically creates a timestamped backup before modifying.',
    schema: { content: z.string().describe('Full content to write to memory') },
    async execute({ content }) {
      const memoryPath = path.join(__dirname, '..', 'memory.md');
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const backupPath = `${memoryPath}.bak.${timestamp}`;
      try {
        await stat(memoryPath);
        await copyFile(memoryPath, backupPath);
      } catch {
        // memory.md doesn't exist yet, skip backup
      }
      await writeFile(memoryPath, String(content), 'utf-8');
      return `Saved to memory.md, backup: ${backupPath}`;
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
