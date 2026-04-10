import { readFile, writeFile, mkdir, readdir, stat } from 'fs/promises';
import path from 'path';
import * as cheerio from 'cheerio';
import { z } from 'zod';

export const readToolSchema = z.object({
  path: z.string().describe('The file path to read'),
});

export const writeToolSchema = z.object({
  path: z.string().describe('The file path to write'),
  content: z.string().describe('The content to write to the file'),
});

export const lsToolSchema = z.object({
  path: z.string().describe('The directory path to list'),
});

export const mkdirToolSchema = z.object({
  path: z.string().describe('The directory path to create'),
});

export const webFetchToolSchema = z.object({
  url: z.string().url().describe('The URL to fetch'),
});

export async function readTool({ path: filePath }: { path: string }): Promise<string> {
  try {
    const content = await readFile(filePath, 'utf-8');
    return content;
  } catch (error) {
    return `Error reading file: ${error instanceof Error ? error.message : String(error)}`;
  }
}

export async function writeTool({ path: filePath, content }: { path: string; content: string }): Promise<string> {
  try {
    await mkdir(path.dirname(filePath), { recursive: true });
    await writeFile(filePath, content, 'utf-8');
    return `Successfully wrote to ${filePath}`;
  } catch (error) {
    return `Error writing file: ${error instanceof Error ? error.message : String(error)}`;
  }
}

export async function lsTool({ path: dirPath }: { path: string }): Promise<string> {
  try {
    const entries = await readdir(dirPath);
    const result = await Promise.all(
      entries.map(async (name) => {
        const fullPath = path.join(dirPath, name);
        const s = await stat(fullPath);
        return `${s.isDirectory() ? 'd' : '-'} ${name}`;
      })
    );
    return result.join('\n');
  } catch (error) {
    return `Error listing directory: ${error instanceof Error ? error.message : String(error)}`;
  }
}

export async function mkdirTool({ path: dirPath }: { path: string }): Promise<string> {
  try {
    await mkdir(dirPath, { recursive: true });
    return `Successfully created directory: ${dirPath}`;
  } catch (error) {
    return `Error creating directory: ${error instanceof Error ? error.message : String(error)}`;
  }
}

export async function webFetchTool({ url }: { url: string }): Promise<string> {
  try {
    const response = await fetch(url, {
      headers: { 'User-Agent': 'AI-Agent/1.0' },
      signal: AbortSignal.timeout(30000),
    });
    const html = await response.text();
    const $ = cheerio.load(html);
    return $('body').text().trim().slice(0, 5000);
  } catch (error) {
    return `Error fetching URL: ${error instanceof Error ? error.message : String(error)}`;
  }
}

export const tools = {
  read: {
    description: 'Read the contents of a file',
    parameters: readToolSchema,
    execute: readTool,
  },
  write: {
    description: 'Write content to a file',
    parameters: writeToolSchema,
    execute: writeTool,
  },
  ls: {
    description: 'List directory contents',
    parameters: lsToolSchema,
    execute: lsTool,
  },
  mkdir: {
    description: 'Create a directory',
    parameters: mkdirToolSchema,
    execute: mkdirTool,
  },
  web_fetch: {
    description: 'Fetch content from a web URL',
    parameters: webFetchToolSchema,
    execute: webFetchTool,
  },
};
