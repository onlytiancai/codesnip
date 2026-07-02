// MiniMax text_to_image API 封装
// 文档：https://platform.minimaxi.com/docs/api-reference/image-generation-t2i
// 指南：https://platform.minimaxi.com/docs/guides/image-generation
//
// 默认使用 response_format: 'url'，按 CLAUDE.md 要求做 URL 解码（%2F → /）。
// 如果想跳过下载步骤，可以传 responseFormat: 'base64'。

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

export const IMAGE_ENDPOINT = 'https://api.minimaxi.com/v1/image_generation';

export type GenerateOptions = {
  prompt: string;
  model?: 'image-01' | 'image-01-live'; // 默认 'image-01'
  aspectRatio?: '1:1' | '16:9' | '4:3' | '3:2' | '2:3' | '3:4' | '9:16' | '21:9';
  width?: number;  // 512~2048，image-01 专用，会被 aspect_ratio 覆盖
  height?: number; // 同上
  responseFormat?: 'url' | 'base64'; // 默认 'url'
  seed?: number;
  promptOptimizer?: boolean;
  n?: number; // 1~9，默认 1
};

export type GeneratedImage = {
  /** 解码后的 URL（仅 url 模式有值） */
  url?: string;
  /** 原始 URL（未解码） */
  rawUrl?: string;
  /** base64 字符串（仅 base64 模式有值） */
  base64?: string;
  /** 任务 ID */
  taskId: string;
};

/**
 * 调用 MiniMax text_to_image，返回生成的图片 URL 或 base64。
 */
export async function generate(opts: GenerateOptions): Promise<GeneratedImage[]> {
  const apiKey = process.env.MINIMAX_API_KEY;
  if (!apiKey) {
    throw new Error('MINIMAX_API_KEY 环境变量未设置');
  }

  const body: Record<string, unknown> = {
    model: opts.model ?? 'image-01',
    prompt: opts.prompt,
    // 默认用 base64：避免 MiniMax 服务端时钟漂移导致签名 URL 过期（曾观察到 403）
    // 如需 url 模式，显式传 responseFormat: 'url'
    response_format: opts.responseFormat ?? 'base64',
    n: opts.n ?? 1,
  };
  if (opts.aspectRatio) body.aspect_ratio = opts.aspectRatio;
  if (opts.width !== undefined) body.width = opts.width;
  if (opts.height !== undefined) body.height = opts.height;
  if (opts.seed !== undefined) body.seed = opts.seed;
  if (opts.promptOptimizer !== undefined) body.prompt_optimizer = opts.promptOptimizer;

  const resp = await fetch(IMAGE_ENDPOINT, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Image HTTP ${resp.status}: ${text}`);
  }

  const json = (await resp.json()) as {
    id?: string;
    data?: {
      image_urls?: string[];
      image_base64?: string[];
    };
    base_resp?: { status_code: number; status_msg: string };
  };

  if (json.base_resp?.status_code !== 0) {
    throw new Error(`Image error ${json.base_resp?.status_code}: ${json.base_resp?.status_msg}`);
  }

  const taskId = json.id ?? 'unknown';
  const out: GeneratedImage[] = [];

  if (json.data?.image_urls) {
    for (const raw of json.data.image_urls) {
      out.push({ rawUrl: raw, url: decodeURIComponent(raw), taskId });
    }
  }
  if (json.data?.image_base64) {
    for (const b64 of json.data.image_base64) {
      out.push({ base64: b64, taskId });
    }
  }

  if (out.length === 0) {
    throw new Error('Image 响应中没有 image_urls / image_base64');
  }

  return out;
}

/**
 * 把 GeneratedImage（url 模式）下载并保存到本地文件，返回绝对路径。
 */
export async function downloadToFile(img: GeneratedImage, outputPath: string): Promise<string> {
  if (!img.url) {
    throw new Error('该 image 不包含 url，无法下载');
  }
  const resp = await fetch(img.url);
  if (!resp.ok) {
    throw new Error(`Download HTTP ${resp.status}`);
  }
  const buffer = Buffer.from(await resp.arrayBuffer());
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, buffer);
  return outputPath;
}

/**
 * 把 GeneratedImage（base64 模式）直接写到本地文件，返回绝对路径。
 */
export function writeBase64ToFile(img: GeneratedImage, outputPath: string): string {
  if (!img.base64) {
    throw new Error('该 image 不包含 base64');
  }
  const buffer = Buffer.from(img.base64, 'base64');
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, buffer);
  return outputPath;
}