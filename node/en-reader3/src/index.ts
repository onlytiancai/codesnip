import { readFile, writeFile, mkdir } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { segmentArticle } from './services/segmenter.js';
import { generateArticleScript, generateScripts } from './services/scriptGenerator.js';
import { generateTTS } from './services/tts.js';
import { evaluateScript, printEvaluationResult } from './services/scriptEvaluator.js';
import { generateSubtitles, getAudioDuration } from './services/subtitleGenerator.js';
import { generateSlide } from './services/slideGenerator.js';
import { generateHtmlSlide } from './services/htmlSlideGenerator.js';
import { captureSlideScreenshot } from './services/browserRecorder.js';
import { extractWordTimings, WordTimingEntry } from './services/wordTimingExtractor.js';
import { assembleSegment, concatenateSegments } from './services/videoAssembler.js';
import { SectionData, ArticleScript, PartType } from './types/index.js';
import { logger } from './utils/logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Validate generated article script before proceeding to processing.
 * Throws an error with details if validation fails.
 */
function validateArticleScript(articleScript: ArticleScript): void {
  const errors: string[] = [];

  // Validate intro
  if (!articleScript.intro.title || articleScript.intro.title.length < 2) {
    errors.push('Intro title is missing or too short');
  }
  if (!articleScript.intro.script || articleScript.intro.script.length < 10) {
    errors.push('Intro script is missing or too short');
  }

  // Validate each segment
  for (const segment of articleScript.segments) {
    // Part 2 (Translation) must have script
    const translationPart = segment.parts.find((p) => p.type === PartType.TRANSLATION);
    if (!translationPart?.script || translationPart.script.length < 10) {
      errors.push(`Segment ${segment.id} Part 2 (Translation): Script is missing or too short`);
    }

    // Part 3 (Vocabulary) - vocabulary array should have items if they were requested
    const vocabPart = segment.parts.find((p) => p.type === PartType.VOCABULARY);
    if (vocabPart && (!vocabPart.vocabulary || vocabPart.vocabulary.length === 0)) {
      logger.warn(`Segment ${segment.id} Part 3 (Vocabulary): No vocabulary items generated`);
    }

    // Part 4 (Grammar) - grammarPoints array should have items if they were requested
    const grammarPart = segment.parts.find((p) => p.type === PartType.GRAMMAR);
    if (grammarPart && (!grammarPart.grammarPoints || grammarPart.grammarPoints.length === 0)) {
      logger.warn(`Segment ${segment.id} Part 4 (Grammar): No grammar points generated`);
    }
  }

  // Validate outro
  if (!articleScript.outro.script || articleScript.outro.script.length < 10) {
    errors.push('Outro script is missing or too short');
  }

  if (errors.length > 0) {
    const errorMsg = `AI content validation failed:\n${errors.map((e) => `  - ${e}`).join('\n')}`;
    logger.error(errorMsg);
    throw new Error(errorMsg);
  }

  logger.info('AI content validation passed');
}

/**
 * Generate video from article text using the new 6-part pipeline.
 */
async function generateVideo(articleText: string, options: { outputPath: string; title?: string }): Promise<string> {
  const title = options.title || 'English Article';
  const outputDir = join(__dirname, '..', 'output');
  const segmentsDir = join(outputDir, 'segments');

  await mkdir(outputDir, { recursive: true });
  await mkdir(segmentsDir, { recursive: true });

  logger.info('Starting video generation pipeline (6-part format)');
  logger.info(`Article length: ${articleText.length} characters`);

  // Step 1: Segment article
  logger.info('Step 1: Segmenting article');
  const segments = segmentArticle(articleText);
  logger.info(`Created ${segments.length} segments`);

  // Step 2: Generate article script with intro, 6-part segments, and outro
  logger.info('Step 2: Generating article script (intro + 6-part segments + outro)');
  const articleScript = await generateArticleScript(segments, title, articleText);
  logger.info(`Generated: 1 intro, ${articleScript.segments.length} segments (6 parts each), 1 outro`);

  // Validate generated content
  validateArticleScript(articleScript);

  // Step 3: Process intro
  logger.info('Step 3: Processing intro');
  const introResult = await processIntro(articleScript.intro, segmentsDir);

  // Step 4: Process all segments with 6 parts each
  logger.info('Step 4: Processing 6-part segments');
  const segmentResults = [];
  for (const segment of articleScript.segments) {
    const segmentParts = await processSegment(segment.id, segment.originalText, segment.parts, segmentsDir);
    segmentResults.push(segmentParts);
  }

  // Step 5: Process outro
  logger.info('Step 5: Processing outro');
  const outroResult = await processOutro(articleScript.outro, segmentsDir);

  // Step 6: Concatenate all parts in order
  logger.info('Step 6: Concatenating all parts into final video');
  const allParts: string[] = [];

  // Add intro
  if (introResult.videoPath) {
    allParts.push(introResult.videoPath);
  }

  // Add segment parts in order
  for (const segmentParts of segmentResults) {
    for (const part of segmentParts) {
      if (part.videoPath) {
        allParts.push(part.videoPath);
      }
    }
  }

  // Add outro
  if (outroResult.videoPath) {
    allParts.push(outroResult.videoPath);
  }

  if (allParts.length === 0) {
    throw new Error('No video parts to concatenate');
  }

  if (allParts.length === 1) {
    // Single part - just rename to output path
    const { rename } = await import('fs/promises');
    await rename(allParts[0], options.outputPath);
  } else {
    // Multiple parts - concatenate
    await concatenateSegments(allParts, options.outputPath);
  }

  logger.info(`Video generation complete: ${options.outputPath}`);
  return options.outputPath;
}

/**
 * Phase 3: Generate slides and audio for all content.
 * Reads from article-script.json, generates TTS audio and HTML slides.
 */
async function runPhase3(
  articleScript: ArticleScript,
  outputDir: string
): Promise<{
  intro: { audioPath: string; subtitlePath: string; htmlPath: string; duration: number };
  segments: { id: number; parts: { partId: number; partType: string; audioPath: string; subtitlePath: string; htmlPath: string; duration: number }[] }[];
  outro: { audioPath: string; subtitlePath: string; htmlPath: string; duration: number };
}> {
  const segmentsDir = join(outputDir, 'segments');
  await mkdir(segmentsDir, { recursive: true });

  const result: {
    intro: { audioPath: string; subtitlePath: string; htmlPath: string; duration: number };
    segments: { id: number; parts: { partId: number; partType: string; audioPath: string; subtitlePath: string; htmlPath: string; duration: number }[] }[];
    outro: { audioPath: string; subtitlePath: string; htmlPath: string; duration: number };
  } = {
    intro: {} as any,
    segments: [],
    outro: {} as any,
  };

  // Process intro
  logger.info('Phase 3: Processing intro...');
  const introDir = join(segmentsDir, 'intro');
  await mkdir(introDir, { recursive: true });
  const introAudio = await generateTTS(articleScript.intro.script, introDir, 0);
  const introSubtitle = await generateSubtitles(articleScript.intro.script, introAudio.audioPath, introDir, 0);

  const introSectionData: SectionData = {
    id: 0,
    originalText: articleScript.intro.title,
    summary: articleScript.intro.script,
    vocabulary: [],
    grammarPoints: [],
    contextExplanation: '',
    narrationScript: articleScript.intro.script,
  };
  const introWordTimings = await extractWordTimings(introSubtitle).catch(() => []);
  const introHtml = await generateHtmlSlide({
    section: introSectionData,
    wordTimings: introWordTimings,
    audioPath: introAudio.audioPath,
    outputDir: introDir,
    partId: 0,
  });

  result.intro = {
    audioPath: introAudio.audioPath,
    subtitlePath: introSubtitle,
    htmlPath: introHtml,
    duration: introAudio.duration,
  };

  // Process segments
  for (const segment of articleScript.segments) {
    logger.info(`Phase 3: Processing segment ${segment.id}...`);
    const segmentDir = join(segmentsDir, `segment-${segment.id}`);
    await mkdir(segmentDir, { recursive: true });

    const segmentResult: typeof result.segments[0] = { id: segment.id, parts: [] };

    for (const part of segment.parts) {
      const partDir = join(segmentDir, `part-${part.id}`);
      await mkdir(partDir, { recursive: true });

      if (!part.script) {
        logger.warn(`No script for segment ${segment.id} part ${part.id}`);
        continue;
      }

      const partAudio = await generateTTS(part.script, partDir, part.id);
      const partSubtitle = await generateSubtitles(part.script, partAudio.audioPath, partDir, part.id);

      const partSectionData: SectionData = {
        id: segment.id,
        originalText: segment.originalText,
        summary: '',
        vocabulary: part.type === PartType.VOCABULARY ? (part.vocabulary || []) : [],
        grammarPoints: part.type === PartType.GRAMMAR ? (part.grammarPoints || []) : [],
        contextExplanation: part.type === PartType.EXPLANATION ? (part.contextExplanation || '') : '',
        narrationScript: part.script,
      };
      const partWordTimings = await extractWordTimings(partSubtitle).catch(() => []);
      const partHtml = await generateHtmlSlide({
        section: partSectionData,
        partType: part.type,
        wordTimings: partWordTimings,
        audioPath: partAudio.audioPath,
        outputDir: partDir,
        partId: part.id,
      });

      segmentResult.parts.push({
        partId: part.id,
        partType: part.type,
        audioPath: partAudio.audioPath,
        subtitlePath: partSubtitle,
        htmlPath: partHtml,
        duration: partAudio.duration,
      });
    }

    result.segments.push(segmentResult);
  }

  // Process outro
  logger.info('Phase 3: Processing outro...');
  const outroDir = join(segmentsDir, 'outro');
  await mkdir(outroDir, { recursive: true });
  const outroAudio = await generateTTS(articleScript.outro.script, outroDir, 999);
  const outroSubtitle = await generateSubtitles(articleScript.outro.script, outroAudio.audioPath, outroDir, 999);

  const outroSectionData: SectionData = {
    id: 999,
    originalText: articleScript.outro.script,
    summary: articleScript.outro.script,
    vocabulary: [],
    grammarPoints: [],
    contextExplanation: '',
    narrationScript: articleScript.outro.script,
  };
  const outroWordTimings = await extractWordTimings(outroSubtitle).catch(() => []);
  const outroHtml = await generateHtmlSlide({
    section: outroSectionData,
    wordTimings: outroWordTimings,
    audioPath: outroAudio.audioPath,
    outputDir: outroDir,
    partId: 999,
  });

  result.outro = {
    audioPath: outroAudio.audioPath,
    subtitlePath: outroSubtitle,
    htmlPath: outroHtml,
    duration: outroAudio.duration,
  };

  return result;
}

/**
 * Phase 4: Generate video segments with screenshots.
 * Reads from phase-3-result.json, captures screenshots and creates MP4 files.
 */
async function runPhase4(
  phase3Result: Awaited<ReturnType<typeof runPhase3>>,
  outputDir: string
): Promise<void> {
  const segmentsDir = join(outputDir, 'segments');

  const phase4Result: {
    intro: { videoPath: string };
    segments: { id: number; parts: { partId: number; videoPath: string }[] }[];
    outro: { videoPath: string };
  } = {
    intro: {} as any,
    segments: [],
    outro: {} as any,
  };

  // Process intro
  logger.info('Phase 4: Processing intro...');
  const introDir = join(segmentsDir, 'intro');
  const introScreenshot = join(introDir, 'slide-intro.png');
  await captureSlideScreenshot({
    htmlPath: phase3Result.intro.htmlPath,
    outputPath: introScreenshot,
    width: 1080,
    height: 1920,
  });
  const introVideoPath = join(introDir, 'intro.mp4');
  const { exec } = await import('child_process');
  await new Promise<void>((resolve, reject) => {
    exec(
      `ffmpeg -y -loop 1 -framerate 25 -i "${introScreenshot}" -i "${phase3Result.intro.audioPath}" -vf "ass='${phase3Result.intro.subtitlePath}'" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -t ${phase3Result.intro.duration} -fps_mode cfr "${introVideoPath}"`,
      (err) => (err ? reject(err) : resolve())
    );
  });
  phase4Result.intro = { videoPath: introVideoPath };

  // Process segments
  for (const segment of phase3Result.segments) {
    logger.info(`Phase 4: Processing segment ${segment.id}...`);
    const segmentDir = join(segmentsDir, `segment-${segment.id}`);
    const segmentResult: typeof phase4Result.segments[0] = { id: segment.id, parts: [] };

    for (const part of segment.parts) {
      const partDir = join(segmentDir, `part-${part.partId}`);
      const screenshotPath = join(partDir, `slide-${segment.id}-${part.partId}.png`);
      await captureSlideScreenshot({
        htmlPath: part.htmlPath,
        outputPath: screenshotPath,
        width: 1080,
        height: 1920,
      });

      const videoPath = join(partDir, `part-${segment.id}-${part.partId}.mp4`);
      await new Promise<void>((resolve, reject) => {
        exec(
          `ffmpeg -y -loop 1 -framerate 25 -i "${screenshotPath}" -i "${part.audioPath}" -vf "ass='${part.subtitlePath}'" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -t ${part.duration} -fps_mode cfr "${videoPath}"`,
          (err) => (err ? reject(err) : resolve())
        );
      });

      segmentResult.parts.push({ partId: part.partId, videoPath });
    }

    phase4Result.segments.push(segmentResult);
  }

  // Process outro
  logger.info('Phase 4: Processing outro...');
  const outroDir = join(segmentsDir, 'outro');
  const outroScreenshot = join(outroDir, 'slide-outro.png');
  await captureSlideScreenshot({
    htmlPath: phase3Result.outro.htmlPath,
    outputPath: outroScreenshot,
    width: 1080,
    height: 1920,
  });
  const outroVideoPath = join(outroDir, 'outro.mp4');
  await new Promise<void>((resolve, reject) => {
    exec(
      `ffmpeg -y -loop 1 -framerate 25 -i "${outroScreenshot}" -i "${phase3Result.outro.audioPath}" -vf "ass='${phase3Result.outro.subtitlePath}'" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -t ${phase3Result.outro.duration} -fps_mode cfr "${outroVideoPath}"`,
      (err) => (err ? reject(err) : resolve())
    );
  });
  phase4Result.outro = { videoPath: outroVideoPath };

  // Save phase 4 result
  const phase4Path = join(outputDir, 'phase-4-result.json');
  await writeFile(phase4Path, JSON.stringify(phase4Result, null, 2), 'utf-8');
}

interface PartResult {
  partId: number;
  partType: PartType;
  audioPath?: string;
  subtitlePath?: string;
  htmlPath?: string;
  screenshotPath?: string;
  videoPath?: string;
  duration?: number;
}

interface IntroResult {
  audioPath?: string;
  subtitlePath?: string;
  htmlPath?: string;
  screenshotPath?: string;
  videoPath?: string;
  duration?: number;
}

interface OutroResult {
  audioPath?: string;
  subtitlePath?: string;
  htmlPath?: string;
  screenshotPath?: string;
  videoPath?: string;
  duration?: number;
}

/**
 * Process intro section.
 */
async function processIntro(
  intro: { title: string; script: string },
  outputDir: string
): Promise<IntroResult> {
  const introDir = join(outputDir, 'intro');
  await mkdir(introDir, { recursive: true });

  const result: IntroResult = {};

  try {
    // Generate TTS for intro
    const { audioPath, duration } = await generateTTS(intro.script, introDir, 0);
    result.audioPath = audioPath;
    result.duration = duration;

    // Generate subtitles
    const subtitlePath = await generateSubtitles(intro.script, audioPath, introDir, 0);
    result.subtitlePath = subtitlePath;

    // Create section data for HTML generation
    const sectionData: SectionData = {
      id: 0,
      originalText: intro.title,
      summary: intro.script,
      vocabulary: [],
      grammarPoints: [],
      contextExplanation: '',
      narrationScript: intro.script,
    };

    // Extract word timings
    let wordTimings: WordTimingEntry[] = [];
    try {
      wordTimings = await extractWordTimings(subtitlePath);
    } catch {
      logger.warn('Failed to extract word timings for intro');
    }

    // Generate HTML slide
    const htmlPath = await generateHtmlSlide({
      section: sectionData,
      wordTimings,
      audioPath,
      outputDir: introDir,
      partId: 0,
    });
    result.htmlPath = htmlPath;

    // Capture screenshot
    const screenshotPath = join(introDir, 'slide-intro.png');
    await captureSlideScreenshot({
      htmlPath,
      outputPath: screenshotPath,
      width: 1080,
      height: 1920,
    });
    result.screenshotPath = screenshotPath;

    // Create video with FFmpeg
    const videoPath = join(introDir, 'intro.mp4');
    const { exec } = await import('child_process');

    await new Promise<void>((resolve, reject) => {
      exec(
        `ffmpeg -y -loop 1 -framerate 25 -i "${screenshotPath}" -i "${audioPath}" -vf "ass='${subtitlePath}'" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -t ${duration} -fps_mode cfr "${videoPath}"`,
        (err) => {
          if (err) reject(err);
          else resolve();
        }
      );
    });
    result.videoPath = videoPath;

    logger.info('Intro processing complete');
  } catch (error) {
    logger.error('Failed to process intro:', error);
  }

  return result;
}

/**
 * Process outro section.
 */
async function processOutro(
  outro: { script: string },
  outputDir: string
): Promise<OutroResult> {
  const outroDir = join(outputDir, 'outro');
  await mkdir(outroDir, { recursive: true });

  const result: OutroResult = {};

  try {
    // Generate TTS for outro
    const { audioPath, duration } = await generateTTS(outro.script, outroDir, 999);
    result.audioPath = audioPath;
    result.duration = duration;

    // Generate subtitles
    const subtitlePath = await generateSubtitles(outro.script, audioPath, outroDir, 999);
    result.subtitlePath = subtitlePath;

    // Create section data for HTML generation
    const sectionData: SectionData = {
      id: 999,
      originalText: outro.script,
      summary: outro.script,
      vocabulary: [],
      grammarPoints: [],
      contextExplanation: '',
      narrationScript: outro.script,
    };

    // Extract word timings
    let wordTimings: WordTimingEntry[] = [];
    try {
      wordTimings = await extractWordTimings(subtitlePath);
    } catch {
      logger.warn('Failed to extract word timings for outro');
    }

    // Generate HTML slide
    const htmlPath = await generateHtmlSlide({
      section: sectionData,
      wordTimings,
      audioPath,
      outputDir: outroDir,
      partId: 999,
    });
    result.htmlPath = htmlPath;

    // Capture screenshot
    const screenshotPath = join(outroDir, 'slide-outro.png');
    await captureSlideScreenshot({
      htmlPath,
      outputPath: screenshotPath,
      width: 1080,
      height: 1920,
    });
    result.screenshotPath = screenshotPath;

    // Create video with FFmpeg
    const videoPath = join(outroDir, 'outro.mp4');
    const { exec } = await import('child_process');

    await new Promise<void>((resolve, reject) => {
      exec(
        `ffmpeg -y -loop 1 -framerate 25 -i "${screenshotPath}" -i "${audioPath}" -vf "ass='${subtitlePath}'" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -t ${duration} -fps_mode cfr "${videoPath}"`,
        (err) => {
          if (err) reject(err);
          else resolve();
        }
      );
    });
    result.videoPath = videoPath;

    logger.info('Outro processing complete');
  } catch (error) {
    logger.error('Failed to process outro:', error);
  }

  return result;
}

/**
 * Process a single segment with 6 parts.
 */
async function processSegment(
  segmentId: number,
  originalText: string,
  parts: { id: number; type: PartType; script?: string }[],
  outputDir: string
): Promise<PartResult[]> {
  const segmentDir = join(outputDir, `segment-${segmentId}`);
  await mkdir(segmentDir, { recursive: true });

  const results: PartResult[] = [];

  for (const part of parts) {
    const partDir = join(segmentDir, `part-${part.id}`);
    await mkdir(partDir, { recursive: true });

    const result: PartResult = {
      partId: part.id,
      partType: part.type,
    };

    try {
      if (!part.script) {
        logger.warn(`No script for segment ${segmentId} part ${part.id}`);
        results.push(result);
        continue;
      }

      // Generate TTS
      const { audioPath, duration } = await generateTTS(part.script, partDir, part.id);
      result.audioPath = audioPath;
      result.duration = duration;

      // Generate subtitles
      const subtitlePath = await generateSubtitles(part.script, audioPath, partDir, part.id);
      result.subtitlePath = subtitlePath;

      // Create section data for HTML generation
      const sectionData: SectionData = {
        id: segmentId,
        originalText,
        summary: '',
        vocabulary: [],
        grammarPoints: [],
        contextExplanation: '',
        narrationScript: part.script,
      };

      // Extract word timings
      let wordTimings: WordTimingEntry[] = [];
      try {
        wordTimings = await extractWordTimings(subtitlePath);
      } catch {
        logger.warn(`Failed to extract word timings for segment ${segmentId} part ${part.id}`);
      }

      // Generate HTML slide
      const htmlPath = await generateHtmlSlide({
        section: sectionData,
        partType: part.type,
        wordTimings,
        audioPath,
        outputDir: partDir,
        partId: part.id,
      });
      result.htmlPath = htmlPath;

      // Capture screenshot
      const screenshotPath = join(partDir, `slide-${segmentId}-${part.id}.png`);
      await captureSlideScreenshot({
        htmlPath,
        outputPath: screenshotPath,
        width: 1080,
        height: 1920,
      });
      result.screenshotPath = screenshotPath;

      // Create video with FFmpeg
      const videoPath = join(partDir, `part-${segmentId}-${part.id}.mp4`);
      const { exec } = await import('child_process');

      await new Promise<void>((resolve, reject) => {
        exec(
          `ffmpeg -y -loop 1 -framerate 25 -i "${screenshotPath}" -i "${audioPath}" -vf "ass='${subtitlePath}'" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -t ${duration} -fps_mode cfr "${videoPath}"`,
          (err) => {
            if (err) reject(err);
            else resolve();
          }
        );
      });
      result.videoPath = videoPath;

      logger.info(`Segment ${segmentId} part ${part.id} (${part.type}) complete`);
    } catch (error) {
      logger.error(`Failed to process segment ${segmentId} part ${part.id}:`, error);
    }

    results.push(result);
  }

  return results;
}

/**
 * Parse command line arguments.
 */
function parseArgs(): { inputPath: string; outputPath: string; title?: string; phase?: number; evaluate?: boolean } {
  const args = process.argv.slice(2);
  let inputPath = '';
  let outputPath = '';
  let title: string | undefined;
  let phase: number | undefined;
  let evaluate = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--input' && i + 1 < args.length) {
      inputPath = args[++i];
    } else if (args[i] === '--output' && i + 1 < args.length) {
      outputPath = args[++i];
    } else if (args[i] === '--title' && i + 1 < args.length) {
      title = args[++i];
    } else if (args[i] === '--phase' && i + 1 < args.length) {
      phase = parseInt(args[++i], 10);
    } else if (args[i] === '--evaluate') {
      evaluate = true;
    }
  }

  return { inputPath, outputPath, title, phase, evaluate };
}

/**
 * Main CLI entry point.
 */
async function main() {
  const { inputPath, outputPath, title, phase, evaluate } = parseArgs();

  try {
    // Handle evaluate mode
    if (evaluate) {
      const scriptPath = join(__dirname, '..', 'output', 'article-script.json');
      const result = await evaluateScript(scriptPath);
      printEvaluationResult(result);
      return;
    }

    const articleText = await readFile(inputPath, 'utf-8');

    // Phase 1: Segment only
    if (phase === 1) {
      logger.info('Phase 1: Segmenting article...');
      const segments = segmentArticle(articleText);
      const segmentsPath = join(__dirname, '..', 'output', 'segments.json');
      await mkdir(join(__dirname, '..', 'output'), { recursive: true });
      await writeFile(segmentsPath, JSON.stringify({ segments }, null, 2), 'utf-8');
      logger.info(`Segmented into ${segments.length} segments, saved to ${segmentsPath}`);
      return;
    }

    // Phase 2: Generate AI scripts (requires phase 1 output)
    if (phase === 2) {
      logger.info('Phase 2: Generating AI scripts...');
      const segmentsPath = join(__dirname, '..', 'output', 'segments.json');
      const segmentsData = await readFile(segmentsPath, 'utf-8');
      const { segments } = JSON.parse(segmentsData);
      const articleTitle = title || 'English Article';
      // Reconstruct original text from segments for intro/outro context
      const originalText = segments.map((s: { text: string }) => s.text).join('\n\n');
      const articleScript = await generateArticleScript(segments, articleTitle, originalText);
      const scriptPath = join(__dirname, '..', 'output', 'article-script.json');
      await writeFile(scriptPath, JSON.stringify(articleScript, null, 2), 'utf-8');
      logger.info(`AI script generated, saved to ${scriptPath}`);
      return;
    }

    // Phase 3: Generate slides & audio (requires phase 2 output)
    if (phase === 3) {
      logger.info('Phase 3: Generating slides and audio...');
      const scriptPath = join(__dirname, '..', 'output', 'article-script.json');
      const scriptData = await readFile(scriptPath, 'utf-8');
      const articleScript: ArticleScript = JSON.parse(scriptData);
      const outputDir = join(__dirname, '..', 'output');

      const phase3Result = await runPhase3(articleScript, outputDir);
      const phase3Path = join(outputDir, 'phase-3-result.json');
      await writeFile(phase3Path, JSON.stringify(phase3Result, null, 2), 'utf-8');
      logger.info(`Phase 3 complete, saved to ${phase3Path}`);
      return;
    }

    // Phase 4: Generate video segments with subtitles (requires phase 3 output)
    if (phase === 4) {
      logger.info('Phase 4: Generating video segments...');
      const phase3Path = join(__dirname, '..', 'output', 'phase-3-result.json');
      const phase3Data = await readFile(phase3Path, 'utf-8');
      const phase3Result = JSON.parse(phase3Data);
      const outputDir = join(__dirname, '..', 'output');

      await runPhase4(phase3Result, outputDir);
      logger.info('Phase 4 complete');
      return;
    }

    // Phase 5: Concatenate all segments into final video (requires phase 4 output)
    if (phase === 5) {
      logger.info('Phase 5: Concatenating segments...');
      if (!outputPath) {
        console.error('Phase 5 requires --output flag');
        process.exit(1);
      }
      const phase4Path = join(__dirname, '..', 'output', 'phase-4-result.json');
      const phase4Data = await readFile(phase4Path, 'utf-8');
      const phase4Result = JSON.parse(phase4Data);

      const allParts: string[] = [];
      if (phase4Result.intro?.videoPath) allParts.push(phase4Result.intro.videoPath);
      for (const seg of phase4Result.segments || []) {
        for (const part of seg.parts || []) {
          if (part.videoPath) allParts.push(part.videoPath);
        }
      }
      if (phase4Result.outro?.videoPath) allParts.push(phase4Result.outro.videoPath);

      if (allParts.length === 0) {
        throw new Error('No video parts to concatenate');
      }

      if (allParts.length === 1) {
        const { rename } = await import('fs/promises');
        await rename(allParts[0], outputPath);
      } else {
        await concatenateSegments(allParts, outputPath);
      }
      logger.info(`Phase 5 complete: ${outputPath}`);
      return;
    }

    // Default: Full pipeline
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
