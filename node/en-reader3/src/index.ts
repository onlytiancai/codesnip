import { readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { mkdir } from 'fs/promises';
import { segmentArticle } from './services/segmenter.js';
import { generateScripts } from './services/scriptGenerator.js';
import { generateTTS } from './services/tts.js';
import { generateSubtitles } from './services/subtitleGenerator.js';
import { generateSlide } from './services/slideGenerator.js';
import { assembleSegment, concatenateSegments } from './services/videoAssembler.js';
import { SectionData, VideoOptions } from './types/index.js';
import { logger } from './utils/logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Generate video from article text.
 */
async function generateVideo(articleText: string, options: VideoOptions): Promise<string> {
  const title = options.title || 'English Article';
  const outputDir = join(__dirname, '..', 'output');
  const segmentsDir = join(outputDir, 'segments');

  await mkdir(outputDir, { recursive: true });
  await mkdir(segmentsDir, { recursive: true });

  logger.info('Starting video generation pipeline');
  logger.info(`Article length: ${articleText.length} characters`);

  // Step 1: Segment article
  logger.info('Step 1: Segmenting article');
  const segments = segmentArticle(articleText);
  logger.info(`Created ${segments.length} segments`);

  // Step 2: Generate AI scripts
  logger.info('Step 2: Generating AI teaching scripts');
  const scripts = await generateScripts(segments, title);

  // Step 3: Generate TTS audio and subtitles
  logger.info('Step 3: Generating TTS audio');
  const audioData: { audioPath: string; duration: number }[] = [];

  for (let i = 0; i < scripts.length; i++) {
    const script = scripts[i];
    const sectionDir = join(segmentsDir, `section-${script.id}`);
    await mkdir(sectionDir, { recursive: true });

    // Save section data JSON
    const sectionData: SectionData = {
      id: script.id,
      originalText: script.originalText,
      summary: script.summary,
      vocabulary: script.vocabulary,
      grammarPoints: script.grammarPoints,
      contextExplanation: script.contextExplanation,
      narrationScript: script.narrationScript,
    };

    const { writeFile } = await import('fs/promises');
    await writeFile(
      join(sectionDir, 'data.json'),
      JSON.stringify(sectionData, null, 2),
      'utf-8'
    );

    // Generate TTS
    const { audioPath, duration } = await generateTTS(
      script.narrationScript,
      sectionDir,
      script.id
    );
    audioData.push({ audioPath, duration });
    scripts[i].audioDuration = duration;
  }

  // Step 4: Generate slides
  logger.info('Step 4: Generating slides');
  const slidePaths: string[] = [];

  for (let i = 0; i < scripts.length; i++) {
    const script = scripts[i];
    const sectionDir = join(segmentsDir, `section-${script.id}`);

    const slideImage = await generateSlide(script, sectionDir);
    slidePaths.push(slideImage);
  }

  // Step 5: Generate subtitles
  logger.info('Step 5: Generating subtitles');
  const subtitlePaths: string[] = [];

  for (let i = 0; i < scripts.length; i++) {
    const script = scripts[i];
    const sectionDir = join(segmentsDir, `section-${script.id}`);
    const { audioPath } = audioData[i];

    const subtitlePath = await generateSubtitles(
      script.narrationScript,
      audioPath,
      sectionDir,
      script.id
    );
    subtitlePaths.push(subtitlePath);
  }

  // Step 6: Assemble video segments
  logger.info('Step 6: Assembling video segments');
  const segmentPaths: string[] = [];

  for (let i = 0; i < scripts.length; i++) {
    const sectionDir = join(segmentsDir, `section-${scripts[i].id}`);

    const { segmentPath } = await assembleSegment(
      slidePaths[i],
      audioData[i].audioPath,
      subtitlePaths[i],
      scripts[i].id,
      sectionDir
    );
    segmentPaths.push(segmentPath);
  }

  // Step 7: Concatenate all segments
  logger.info('Step 7: Concatenating segments');
  await concatenateSegments(segmentPaths, options.outputPath);

  logger.info(`Video generation complete: ${options.outputPath}`);

  return options.outputPath;
}

/**
 * Parse command line arguments.
 */
function parseArgs(): { inputPath: string; outputPath: string; title?: string } {
  const args = process.argv.slice(2);
  let inputPath = '';
  let outputPath = '';
  let title: string | undefined;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--input' && i + 1 < args.length) {
      inputPath = args[++i];
    } else if (args[i] === '--output' && i + 1 < args.length) {
      outputPath = args[++i];
    } else if (args[i] === '--title' && i + 1 < args.length) {
      title = args[++i];
    }
  }

  if (!inputPath || !outputPath) {
    console.error('Usage: tsx src/index.ts --input <input.txt> --output <output.mp4> [--title "Title"]');
    process.exit(1);
  }

  return { inputPath, outputPath, title };
}

/**
 * Main CLI entry point.
 */
async function main() {
  const { inputPath, outputPath, title } = parseArgs();

  try {
    const articleText = await readFile(inputPath, 'utf-8');

    await generateVideo(articleText, {
      outputPath,
      title,
    });

    logger.info('Done!');
  } catch (error) {
    logger.error('Video generation failed:', error);
    process.exit(1);
  }
}

main();
