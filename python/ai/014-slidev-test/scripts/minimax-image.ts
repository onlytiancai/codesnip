// MiniMax text_to_image API 封装
// 文档：https://platform.minimaxi.com/docs/api-reference/image-generation-t2i

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

export const IMAGE_ENDPOINT = 'https://api.minimaxi.com/v1/image_generation';

export type GenerateOptions = {
  prompt: string;
  model?: 'image-01' | 'image-01-live';
  aspectRatio?: '1:1' | '16:9' | '4:3' | '3:2' | '2:3' | '3:4' | '9:16' | '21:9';
  width?: number;
  height?: number;
  responseFormat?: 'url' | 'base64';
  seed?: number;
  promptOptimizer?: boolean;
  n?: number;
};

export type GeneratedImage = {
  url?: string;
  rawUrl?: string;
  base64?: string;
  taskId: string;
};

export async function generate(opts: GenerateOptions): Promise<GeneratedImage[]> {
  const apiKey = process.env.MINIMAX_API_KEY;
  if (!apiKey) {
    throw new Error('MINIMAX_API_KEY 环境变量未设置');
  }

  const body: Record<string, unknown> = {
    model: opts.model ?? 'image-01',
    prompt: opts.prompt,
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

export function writeBase64ToFile(img: GeneratedImage, outputPath: string): string {
  if (!img.base64) {
    throw new Error('该 image 不包含 base64');
  }
  const buffer = Buffer.from(img.base64, 'base64');
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, buffer);
  return outputPath;
}
