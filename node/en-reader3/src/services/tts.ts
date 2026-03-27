import { spawn } from 'child_process';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { config, TTSProvider } from '../config/index.js';
import { logger } from '../utils/logger.js';

interface TTSResult {
  audioPath: string;
  duration: number;
}

/**
 * Preprocess narration script by removing [朗读] and [讲解] markers.
 * These markers are for internal word timing but should not be spoken by TTS.
 */
function preprocessNarrationScript(text: string): string {
  // Remove [朗读] and [讲解] markers but keep the content
  let clean = text.replace(/\[朗读\]/g, '');
  clean = clean.replace(/\[讲解\]/g, '');

  // Clean up any extra whitespace left behind
  clean = clean.replace(/\s+/g, ' ').trim();

  return clean;
}

/**
 * Generate Chinese TTS audio using Edge TTS (free, supports Chinese).
 */
async function generateEdgeTTS(text: string, outputPath: string): Promise<TTSResult> {
  logger.debug(`Generating Edge TTS: ${text.substring(0, 50)}...`);

  // Write text to a temp file to avoid CLI argument parsing issues with Chinese characters
  const { writeFile } = await import('fs/promises');
  const { join } = await import('path');
  const { tmpdir } = await import('os');

  const textFile = join(tmpdir(), `tts-text-${Date.now()}.txt`);
  await writeFile(textFile, text, 'utf-8');

  return new Promise((resolve, reject) => {
    // Use shell mode with proper quoting to handle Chinese characters
    const cmd = `python3 -m edge_tts -f "${textFile}" -v zh-CN-XiaoxiaoNeural --write-media "${outputPath}"`;

    const proc = spawn(cmd, {
      shell: true,
    });

    let stderr = '';

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', async (code) => {
      // Clean up temp file
      try {
        const { unlink } = await import('fs/promises');
        await unlink(textFile);
      } catch {
        // Ignore cleanup errors
      }

      if (code === 0) {
        // Get duration using ffprobe
        const duration = await getAudioDuration(outputPath);
        resolve({ audioPath: outputPath, duration });
      } else {
        reject(new Error(`Edge TTS failed: ${stderr}`));
      }
    });

    proc.on('error', (err) => {
      reject(err);
    });
  });
}

/**
 * Generate Chinese TTS audio using ByteDance API.
 */
async function generateByteDanceTTS(text: string, outputPath: string): Promise<TTSResult> {
  logger.debug(`Generating ByteDance TTS: ${text.substring(0, 50)}...`);

  const { BYTEDANCE_APP_ID, BYTEDANCE_ACCESS_KEY, BYTEDANCE_RESOURCE_ID } = config;

  if (!BYTEDANCE_APP_ID || !BYTEDANCE_ACCESS_KEY || !BYTEDANCE_RESOURCE_ID) {
    throw new Error('ByteDance TTS credentials not configured');
  }

  try {
    const response = await fetch(
      'https://openspeech.bytedance.com/api/v3/tts/unidirectional',
      {
        method: 'POST',
        headers: {
          'X-Api-App-Id': BYTEDANCE_APP_ID,
          'X-Api-Access-Key': BYTEDANCE_ACCESS_KEY,
          'X-Api-Resource-Id': BYTEDANCE_RESOURCE_ID,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user: { uid: 'en-reader3' },
          req_params: {
            text: text,
            speaker: 'zh_female_cancan_mars_bigtts',
            audio_params: {
              format: 'mp3',
              sample_rate: 24000,
              enable_timestamp: true,
            },
            additions: JSON.stringify({
              explicit_language: 'zh',
              disable_markdown_filter: true,
            }),
          },
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`ByteDance API error: ${response.status}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    await writeFile(outputPath, buffer);

    const duration = await getAudioDuration(outputPath);

    return { audioPath: outputPath, duration };
  } catch (error) {
    logger.error('ByteDance TTS error:', error);
    throw error;
  }
}

/**
 * Get audio duration using ffprobe.
 */
async function getAudioDuration(audioPath: string): Promise<number> {
  const { spawn } = await import('child_process');

  return new Promise((resolve, reject) => {
    const args = [
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=noprint_wrappers=1:nokey=1',
      audioPath,
    ];

    const proc = spawn('ffprobe', args);

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      if (code === 0) {
        const duration = parseFloat(stdout.trim());
        resolve(isNaN(duration) ? 0 : duration);
      } else {
        reject(new Error(`ffprobe failed: ${stderr}`));
      }
    });
  });
}

/**
 * Generate TTS audio for narration script.
 */
export async function generateTTS(
  text: string,
  outputDir: string,
  sectionId: number
): Promise<TTSResult> {
  await mkdir(outputDir, { recursive: true });

  // Preprocess narration script: remove [朗读] and [讲解] markers
  // These markers are for internal use but should not be spoken by TTS
  const cleanText = preprocessNarrationScript(text);
  logger.debug(`Clean narration: ${cleanText.substring(0, 50)}...`);

  const outputPath = join(outputDir, `section-${sectionId}.mp3`);

  const provider = config.TTS_PROVIDER as TTSProvider;

  switch (provider) {
    case 'bytedance':
      return generateByteDanceTTS(cleanText, outputPath);
    case 'edge':
    default:
      return generateEdgeTTS(cleanText, outputPath);
  }
}
