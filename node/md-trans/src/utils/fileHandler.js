import { readFile, writeFile } from 'fs/promises';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

export async function readMarkdown(filePath) {
  const absolutePath = resolve(filePath);
  const content = await readFile(absolutePath, 'utf-8');
  return content;
}

export async function writeMarkdown(filePath, content) {
  const absolutePath = resolve(filePath);
  await writeFile(absolutePath, content, 'utf-8');
  return absolutePath;
}

export async function fileExists(filePath) {
  try {
    const absolutePath = resolve(filePath);
    await readFile(absolutePath);
    return true;
  } catch {
    return false;
  }
}

export function resolvePath(filePath, baseDir = process.cwd()) {
  return resolve(baseDir, filePath);
}

export default {
  readMarkdown,
  writeMarkdown,
  fileExists,
  resolvePath,
};
