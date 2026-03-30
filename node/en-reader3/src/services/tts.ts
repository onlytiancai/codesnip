import { spawn } from 'child_process';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { config, TTSProvider } from '../config/index.js';
import { logger } from '../utils/logger.js';

interface TTSResult {
  audioPath: string;
  duration: number;
}

interface WordTiming {
  text: string;
  start: number;
  end: number;
}

interface TTSResultWithWords extends TTSResult {
  wordTimings: WordTiming[];
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
 * This function generates audio AND returns word timings in a SINGLE call.
 */
async function generateEdgeTTS(text: string, outputPath: string): Promise<TTSResult> {
  const result = await generateEdgeTTSWithWords(text, outputPath);
  return { audioPath: result.audioPath, duration: result.duration };
}

/**
 * Generate Chinese TTS audio using Edge TTS AND get word timings simultaneously.
 * This uses a single Python script that captures both audio and WordBoundary.
 */
async function generateEdgeTTSWithWords(text: string, outputPath: string): Promise<TTSResultWithWords> {
  logger.debug(`Generating Edge TTS with word timings: ${text.substring(0, 50)}...`);

  // Escape text for Python string
  const escapedText = text.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/`/g, '\\`');

  // Python script that generates audio AND captures WordBoundary in one call
  const pythonScript = `
import sys
import asyncio
import edge_tts

async def generate_and_get_timings(text, output_path):
    words = []
    audio_chunks = []

    communicate = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural", boundary="WordBoundary")

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            offset_seconds = chunk["offset"] / 10000000
            duration_seconds = chunk["duration"] / 10000000
            words.append({
                "text": chunk["text"],
                "start": round(offset_seconds, 3),
                "end": round(offset_seconds + duration_seconds, 3)
            })

    # Write audio file
    with open(output_path, "wb") as f:
        for chunk in audio_chunks:
            f.write(chunk)

    return words

if __name__ == "__main__":
    text = """${escapedText}"""
    output_path = """${outputPath.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"""

    words = asyncio.run(generate_and_get_timings(text, output_path))

    # Output words as JSON for parsing
    import json
    print("WORDS_START")
    for w in words:
        print(f"{w['start']:.3f}:{w['end']:.3f}:{w['text']}")
    print("WORDS_END")
    print(f"DURATION:{len(words)} words")
`;

  return new Promise((resolve, reject) => {
    const proc = spawn('python3', ['-c', pythonScript], { shell: false });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => { stdout += data.toString(); });
    proc.stderr.on('data', (data) => { stderr += data.toString(); });

    proc.on('close', async (code) => {
      if (code === 0) {
        try {
          // Parse word timings from stdout
          const lines = stdout.split('\n');
          const wordTimings: WordTiming[] = [];
          let inWords = false;

          for (const line of lines) {
            if (line === 'WORDS_START') {
              inWords = true;
              continue;
            }
            if (line === 'WORDS_END') {
              inWords = false;
              continue;
            }
            if (line.startsWith('DURATION:')) {
              continue;
            }
            if (inWords && line.trim()) {
              const parts = line.split(':');
              if (parts.length >= 3) {
                wordTimings.push({
                  start: parseFloat(parts[0]),
                  end: parseFloat(parts[1]),
                  text: parts.slice(2).join(':'), // Handle text with colons
                });
              }
            }
          }

          const duration = await getAudioDuration(outputPath);
          logger.debug(`Edge TTS generated ${wordTimings.length} words, audio duration: ${duration}s`);

          resolve({
            audioPath: outputPath,
            duration,
            wordTimings,
          });
        } catch (err) {
          reject(err);
        }
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

/**
 * Generate TTS audio for narration script AND get word timings simultaneously.
 * Only works for Edge TTS provider.
 */
export async function generateTTSWithWords(
  text: string,
  outputDir: string,
  sectionId: number
): Promise<TTSResultWithWords> {
  await mkdir(outputDir, { recursive: true });

  // Preprocess narration script: remove [朗读] and [讲解] markers
  const cleanText = preprocessNarrationScript(text);
  logger.debug(`Clean narration: ${cleanText.substring(0, 50)}...`);

  const outputPath = join(outputDir, `section-${sectionId}.mp3`);

  const provider = config.TTS_PROVIDER as TTSProvider;

  switch (provider) {
    case 'edge':
      return generateEdgeTTSWithWords(cleanText, outputPath);
    case 'bytedance':
      // ByteDance doesn't support simultaneous word timings, use regular TTS
      const result = await generateByteDanceTTS(cleanText, outputPath);
      return { ...result, wordTimings: [] };
    default:
      const defaultResult = await generateEdgeTTS(cleanText, outputPath);
      return { ...defaultResult, wordTimings: [] };
  }
}
