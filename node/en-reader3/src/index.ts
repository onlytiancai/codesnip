import { readFile, writeFile, mkdir, access } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { segmentArticle } from './services/segmenter.js';
import { generateArticleScript, generateScripts } from './services/scriptGenerator.js';
import { generateTTS, generateTTSWithWords } from './services/tts.js';
import { evaluateScript, printEvaluationResult } from './services/scriptEvaluator.js';
import { generateSubtitles, getAudioDuration } from './services/subtitleGenerator.js';
import { generateSlide } from './services/slideGenerator.js';
import { generateHtmlSlide } from './services/htmlSlideGenerator.js';
import { captureSlideScreenshot } from './services/browserRecorder.js';
import { extractWordTimings, WordTimingEntry } from './services/wordTimingExtractor.js';
import { assembleSegment, concatenateSegments } from './services/videoAssembler.js';
import { checkASSFiles, printASSCheckResults } from './services/assChecker.js';
import { startPreviewServer } from './services/previewServer.js';
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
 * Phase 3: Generate TTS audio for all content.
 * Reads from article-script.json, generates TTS audio only (HTML generation moved to phase 5).
 */
async function runPhase3(
  articleScript: ArticleScript,
  outputDir: string
): Promise<{
  intro: { audioPath: string; duration: number; script: string };
  segments: { id: number; parts: { partId: number; partType: string; audioPath: string; duration: number; script: string }[] }[];
  outro: { audioPath: string; duration: number; script: string };
}> {
  const segmentsDir = join(outputDir, 'segments');
  await mkdir(segmentsDir, { recursive: true });

  const result: {
    intro: { audioPath: string; duration: number; script: string };
    segments: { id: number; parts: { partId: number; partType: string; audioPath: string; duration: number; script: string }[] }[];
    outro: { audioPath: string; duration: number; script: string };
  } = {
    intro: {} as any,
    segments: [],
    outro: {} as any,
  };

  // Process intro
  logger.info('Phase 3: Processing intro...');
  const introDir = join(segmentsDir, 'intro');
  await mkdir(introDir, { recursive: true });
  const introTTS = await generateTTSWithWords(articleScript.intro.script, introDir, 0);

  result.intro = {
    audioPath: introTTS.audioPath,
    duration: introTTS.duration,
    script: articleScript.intro.script,
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

      const partTTS = await generateTTSWithWords(part.script, partDir, part.id);

      segmentResult.parts.push({
        partId: part.id,
        partType: part.type,
        audioPath: partTTS.audioPath,
        duration: partTTS.duration,
        script: part.script,
      });
    }

    result.segments.push(segmentResult);
  }

  // Process outro
  logger.info('Phase 3: Processing outro...');
  const outroDir = join(segmentsDir, 'outro');
  await mkdir(outroDir, { recursive: true });
  const outroTTS = await generateTTSWithWords(articleScript.outro.script, outroDir, 999);

  result.outro = {
    audioPath: outroTTS.audioPath,
    duration: outroTTS.duration,
    script: articleScript.outro.script,
  };

  return result;
}

/**
 * Phase 4: Generate ASS subtitle files.
 * Reads from phase-3-result.json, generates ASS for intro, segments, and outro.
 */
async function runPhase4(
  phase3Result: Awaited<ReturnType<typeof runPhase3>>,
  outputDir: string
): Promise<void> {
  const segmentsDir = join(outputDir, 'segments');

  const phase4Result: {
    intro: { subtitlePath: string };
    segments: { id: number; parts: { partId: number; subtitlePath: string }[] }[];
    outro: { subtitlePath: string };
  } = {
    intro: {} as any,
    segments: [],
    outro: {} as any,
  };

  // Process intro
  logger.info('Phase 4: Processing intro...');
  const introDir = join(segmentsDir, 'intro');
  const introSubtitlePath = await generateSubtitles(
    phase3Result.intro.script,
    phase3Result.intro.audioPath,
    introDir,
    0
  );
  phase4Result.intro = { subtitlePath: introSubtitlePath };

  // Process segments
  for (const segment of phase3Result.segments) {
    logger.info(`Phase 4: Processing segment ${segment.id}...`);
    const segmentDir = join(segmentsDir, `segment-${segment.id}`);
    const segmentResult: typeof phase4Result.segments[0] = { id: segment.id, parts: [] };

    for (const part of segment.parts) {
      const partDir = join(segmentDir, `part-${part.partId}`);
      const subtitlePath = await generateSubtitles(
        part.script,
        part.audioPath,
        partDir,
        part.partId
      );
      segmentResult.parts.push({ partId: part.partId, subtitlePath });
    }

    phase4Result.segments.push(segmentResult);
  }

  // Process outro
  logger.info('Phase 4: Processing outro...');
  const outroDir = join(segmentsDir, 'outro');
  const outroSubtitlePath = await generateSubtitles(
    phase3Result.outro.script,
    phase3Result.outro.audioPath,
    outroDir,
    999
  );
  phase4Result.outro = { subtitlePath: outroSubtitlePath };

  // Save phase 4 result
  const phase4Path = join(outputDir, 'phase-4-result.json');
  await writeFile(phase4Path, JSON.stringify(phase4Result, null, 2), 'utf-8');
}

/**
 * Phase 5: Generate PNG screenshots and MP4 videos for each segment.
 * Runs checkASS first; exits with error if validation fails.
 */
async function runPhase5(
  phase3Result: Awaited<ReturnType<typeof runPhase3>>,
  phase4Result: { intro: { subtitlePath: string }; segments: { id: number; parts: { partId: number; subtitlePath: string }[] }[]; outro: { subtitlePath: string } },
  articleScript: ArticleScript,
  outputDir: string
): Promise<void> {
  const segmentsDir = join(outputDir, 'segments');

  // Run checkASS before generating videos
  logger.info('Phase 5: Running ASS validation...');
  const checkResults = await checkASSFiles(segmentsDir);
  const hasErrors = checkResults.results.some((r: { result: { errors: unknown[] } }) => r.result.errors.length > 0);
  if (hasErrors) {
    printASSCheckResults(segmentsDir, checkResults);
    throw new Error('Phase 5: ASS validation failed. Fix errors before proceeding.');
  }
  logger.info('Phase 5: ASS validation passed.');

  const { exec } = await import('child_process');

  const phase5Result: {
    intro: { videoPath: string };
    segments: { id: number; parts: { partId: number; videoPath: string }[] }[];
    outro: { videoPath: string };
  } = {
    intro: {} as any,
    segments: [],
    outro: {} as any,
  };

  // Process intro - generate HTML first
  logger.info('Phase 5: Processing intro...');
  const introDir = join(segmentsDir, 'intro');
  
  // Extract word timings from subtitle
  let introWordTimings: WordTimingEntry[] = [];
  try {
    introWordTimings = await extractWordTimings(phase4Result.intro.subtitlePath);
  } catch {
    logger.warn('Failed to extract word timings for intro');
  }
  
  const introSectionData: SectionData = {
    id: 0,
    originalText: articleScript.intro.title,
    summary: articleScript.intro.script,
    vocabulary: [],
    grammarPoints: [],
    contextExplanation: '',
    narrationScript: articleScript.intro.script,
  };
  const introHtmlPath = await generateHtmlSlide({
    section: introSectionData,
    wordTimings: introWordTimings.map(w => ({ word: w.word, start: w.start, end: w.end })),
    audioPath: phase3Result.intro.audioPath,
    outputDir: introDir,
    partId: 0,
  });

  const introScreenshot = join(introDir, 'slide-intro.png');
  await captureSlideScreenshot({
    htmlPath: introHtmlPath,
    outputPath: introScreenshot,
    width: 1080,
    height: 1920,
  });
  const introVideoPath = join(introDir, 'intro.mp4');
  await new Promise<void>((resolve, reject) => {
    exec(
      `ffmpeg -y -loop 1 -framerate 25 -i "${introScreenshot}" -i "${phase3Result.intro.audioPath}" -vf "ass='${phase4Result.intro.subtitlePath}'" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -t ${phase3Result.intro.duration} -fps_mode cfr "${introVideoPath}"`,
      (err) => (err ? reject(err) : resolve())
    );
  });
  phase5Result.intro = { videoPath: introVideoPath };

  // Process segments
  for (let i = 0; i < phase3Result.segments.length; i++) {
    const segment = phase3Result.segments[i];
    const phase4Segment = phase4Result.segments[i];
    const articleSegment = articleScript.segments[i];
    logger.info(`Phase 5: Processing segment ${segment.id}...`);
    const segmentDir = join(segmentsDir, `segment-${segment.id}`);
    const segmentResult: typeof phase5Result.segments[0] = { id: segment.id, parts: [] };

    for (let j = 0; j < segment.parts.length; j++) {
      const part = segment.parts[j];
      const phase4Part = phase4Segment.parts[j];
      const articlePart = articleSegment.parts[j];
      const partDir = join(segmentDir, `part-${part.partId}`);

      // Extract word timings from subtitle
      let partWordTimings: WordTimingEntry[] = [];
      try {
        partWordTimings = await extractWordTimings(phase4Part.subtitlePath);
      } catch {
        logger.warn(`Failed to extract word timings for segment ${segment.id} part ${part.partId}`);
      }

      // Generate HTML
      const partSectionData: SectionData = {
        id: segment.id,
        originalText: articleSegment.originalText,
        summary: '',
        vocabulary: part.partType === PartType.VOCABULARY ? (articlePart.vocabulary || []) : [],
        grammarPoints: part.partType === PartType.GRAMMAR ? (articlePart.grammarPoints || []) : [],
        contextExplanation: part.partType === PartType.EXPLANATION ? (articlePart.contextExplanation || '') : '',
        narrationScript: part.script,
      };
      const partHtmlPath = await generateHtmlSlide({
        section: partSectionData,
        partType: part.partType as PartType,
        wordTimings: partWordTimings.map(w => ({ word: w.word, start: w.start, end: w.end })),
        audioPath: part.audioPath,
        outputDir: partDir,
        partId: part.partId,
      });

      const screenshotPath = join(partDir, `slide-${segment.id}-${part.partId}.png`);
      await captureSlideScreenshot({
        htmlPath: partHtmlPath,
        outputPath: screenshotPath,
        width: 1080,
        height: 1920,
      });

      const videoPath = join(partDir, `part-${segment.id}-${part.partId}.mp4`);
      await new Promise<void>((resolve, reject) => {
        exec(
          `ffmpeg -y -loop 1 -framerate 25 -i "${screenshotPath}" -i "${part.audioPath}" -vf "ass='${phase4Part.subtitlePath}'" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -t ${part.duration} -fps_mode cfr "${videoPath}"`,
          (err) => (err ? reject(err) : resolve())
        );
      });

      segmentResult.parts.push({ partId: part.partId, videoPath });
    }

    phase5Result.segments.push(segmentResult);
  }

  // Process outro - generate HTML first
  logger.info('Phase 5: Processing outro...');
  const outroDir = join(segmentsDir, 'outro');
  
  // Extract word timings from subtitle
  let outroWordTimings: WordTimingEntry[] = [];
  try {
    outroWordTimings = await extractWordTimings(phase4Result.outro.subtitlePath);
  } catch {
    logger.warn('Failed to extract word timings for outro');
  }
  
  const outroSectionData: SectionData = {
    id: 999,
    originalText: articleScript.outro.script,
    summary: articleScript.outro.script,
    vocabulary: [],
    grammarPoints: [],
    contextExplanation: '',
    narrationScript: articleScript.outro.script,
  };
  const outroHtmlPath = await generateHtmlSlide({
    section: outroSectionData,
    wordTimings: outroWordTimings.map(w => ({ word: w.word, start: w.start, end: w.end })),
    audioPath: phase3Result.outro.audioPath,
    outputDir: outroDir,
    partId: 999,
  });

  const outroScreenshot = join(outroDir, 'slide-outro.png');
  await captureSlideScreenshot({
    htmlPath: outroHtmlPath,
    outputPath: outroScreenshot,
    width: 1080,
    height: 1920,
  });
  const outroVideoPath = join(outroDir, 'outro.mp4');
  await new Promise<void>((resolve, reject) => {
    exec(
      `ffmpeg -y -loop 1 -framerate 25 -i "${outroScreenshot}" -i "${phase3Result.outro.audioPath}" -vf "ass='${phase4Result.outro.subtitlePath}'" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -t ${phase3Result.outro.duration} -fps_mode cfr "${outroVideoPath}"`,
      (err) => (err ? reject(err) : resolve())
    );
  });
  phase5Result.outro = { videoPath: outroVideoPath };

  // Save phase 5 result
  const phase5Path = join(outputDir, 'phase-5-result.json');
  await writeFile(phase5Path, JSON.stringify(phase5Result, null, 2), 'utf-8');
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
function parseArgs(): { inputPath: string; outputPath: string; title?: string; phase?: number; evaluate?: boolean; checkAss?: boolean; previewHtml?: boolean } {
  const args = process.argv.slice(2);
  let inputPath = '';
  let outputPath = '';
  let title: string | undefined;
  let phase: number | undefined;
  let evaluate = false;
  let checkAss = false;
  let previewHtml = false;

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
    } else if (args[i] === '--check-ass') {
      checkAss = true;
    } else if (args[i] === '--preview-html') {
      previewHtml = true;
    }
  }

  return { inputPath, outputPath, title, phase, evaluate, checkAss, previewHtml };
}

/**
 * Check if a file exists and is readable.
 */
async function fileExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

/**
 * Validate phase 1 input: input file must exist and be readable.
 */
async function validatePhase1(inputPath: string): Promise<void> {
  if (!inputPath) {
    throw new Error('Phase 1 requires --input flag');
  }
  if (!(await fileExists(inputPath))) {
    throw new Error(`Input file does not exist or is not readable: ${inputPath}`);
  }
}

/**
 * Validate phase 2 input: segments.json must exist and have valid structure.
 */
async function validatePhase2(): Promise<void> {
  const segmentsPath = join(__dirname, '..', 'output', 'segments.json');
  if (!(await fileExists(segmentsPath))) {
    throw new Error(`Phase 2 requires ${segmentsPath} (run phase 1 first)`);
  }
  const data = JSON.parse(await readFile(segmentsPath, 'utf-8'));
  if (!data.segments || !Array.isArray(data.segments)) {
    throw new Error(`Invalid segments.json structure: missing or invalid segments array`);
  }
}

/**
 * Validate phase 3 input: article-script.json must exist and have valid structure.
 */
async function validatePhase3(): Promise<void> {
  const scriptPath = join(__dirname, '..', 'output', 'article-script.json');
  if (!(await fileExists(scriptPath))) {
    throw new Error(`Phase 3 requires ${scriptPath} (run phase 2 first)`);
  }
  const data = JSON.parse(await readFile(scriptPath, 'utf-8'));
  if (!data.intro || !data.segments || !data.outro) {
    throw new Error(`Invalid article-script.json structure: missing intro, segments, or outro`);
  }
}

/**
 * Validate phase 4 input: phase-3-result.json must exist with audio files.
 * Phase 4 (ASS subtitles) needs audio files from phase 3.
 */
async function validatePhase4(): Promise<void> {
  const phase3Path = join(__dirname, '..', 'output', 'phase-3-result.json');
  if (!(await fileExists(phase3Path))) {
    throw new Error(`Phase 4 requires ${phase3Path} (run phase 3 first)`);
  }
  const data = JSON.parse(await readFile(phase3Path, 'utf-8'));

  // Validate intro files (audio only)
  if (data.intro) {
    if (data.intro.audioPath && !(await fileExists(data.intro.audioPath))) {
      throw new Error(`Phase 4 missing audio file: ${data.intro.audioPath}`);
    }
  }

  // Validate segment files
  for (const seg of data.segments || []) {
    for (const part of seg.parts || []) {
      if (part.audioPath && !(await fileExists(part.audioPath))) {
        throw new Error(`Phase 4 missing audio file: ${part.audioPath}`);
      }
    }
  }

  // Validate outro files
  if (data.outro) {
    if (data.outro.audioPath && !(await fileExists(data.outro.audioPath))) {
      throw new Error(`Phase 4 missing audio file: ${data.outro.audioPath}`);
    }
  }
}

/**
 * Validate phase 5 input: phase-3-result.json, phase-4-result.json and article-script.json must exist.
 * Phase 5 generates HTML, PNG + MP4 from audio and subtitle files.
 */
async function validatePhase5Input(): Promise<void> {
  const phase3Path = join(__dirname, '..', 'output', 'phase-3-result.json');
  if (!(await fileExists(phase3Path))) {
    throw new Error(`Phase 5 requires ${phase3Path} (run phase 3 first)`);
  }
  const phase4Path = join(__dirname, '..', 'output', 'phase-4-result.json');
  if (!(await fileExists(phase4Path))) {
    throw new Error(`Phase 5 requires ${phase4Path} (run phase 4 first)`);
  }
  const articleScriptPath = join(__dirname, '..', 'output', 'article-script.json');
  if (!(await fileExists(articleScriptPath))) {
    throw new Error(`Phase 5 requires ${articleScriptPath} (run phase 2 first)`);
  }

  // Validate phase 3 audio files exist
  const phase3Data = JSON.parse(await readFile(phase3Path, 'utf-8'));
  if (phase3Data.intro) {
    if (phase3Data.intro.audioPath && !(await fileExists(phase3Data.intro.audioPath))) {
      throw new Error(`Phase 5 missing audio file: ${phase3Data.intro.audioPath}`);
    }
  }
  for (const seg of phase3Data.segments || []) {
    for (const part of seg.parts || []) {
      if (part.audioPath && !(await fileExists(part.audioPath))) {
        throw new Error(`Phase 5 missing audio file: ${part.audioPath}`);
      }
    }
  }
  if (phase3Data.outro) {
    if (phase3Data.outro.audioPath && !(await fileExists(phase3Data.outro.audioPath))) {
      throw new Error(`Phase 5 missing audio file: ${phase3Data.outro.audioPath}`);
    }
  }

  // Validate phase 4 subtitle files exist
  const phase4Data = JSON.parse(await readFile(phase4Path, 'utf-8'));
  if (phase4Data.intro?.subtitlePath && !(await fileExists(phase4Data.intro.subtitlePath))) {
    throw new Error(`Phase 5 missing subtitle file: ${phase4Data.intro.subtitlePath}`);
  }
  for (const seg of phase4Data.segments || []) {
    for (const part of seg.parts || []) {
      if (part.subtitlePath && !(await fileExists(part.subtitlePath))) {
        throw new Error(`Phase 5 missing subtitle file: ${part.subtitlePath}`);
      }
    }
  }
  if (phase4Data.outro?.subtitlePath && !(await fileExists(phase4Data.outro.subtitlePath))) {
    throw new Error(`Phase 5 missing subtitle file: ${phase4Data.outro.subtitlePath}`);
  }
}

/**
 * Validate phase 6 input: phase-5-result.json must exist with all referenced MP4 files.
 * Phase 6 (concatenation) needs MP4 video files from phase 5.
 */
async function validatePhase6Input(): Promise<void> {
  const phase5Path = join(__dirname, '..', 'output', 'phase-5-result.json');
  if (!(await fileExists(phase5Path))) {
    throw new Error(`Phase 6 requires ${phase5Path} (run phase 5 first)`);
  }
  const data = JSON.parse(await readFile(phase5Path, 'utf-8'));

  // Collect all MP4 paths
  const mp4Paths: string[] = [];
  if (data.intro?.videoPath) mp4Paths.push(data.intro.videoPath);
  for (const seg of data.segments || []) {
    for (const part of seg.parts || []) {
      if (part.videoPath) mp4Paths.push(part.videoPath);
    }
  }
  if (data.outro?.videoPath) mp4Paths.push(data.outro.videoPath);

  for (const mp4Path of mp4Paths) {
    if (!(await fileExists(mp4Path))) {
      throw new Error(`Phase 6 missing video file: ${mp4Path}`);
    }
  }
}

/**
 * Main CLI entry point.
 */
async function main() {
  const { inputPath, outputPath, title, phase, evaluate, checkAss, previewHtml } = parseArgs();

  try {
    // Handle preview-html mode
    if (previewHtml) {
      const scriptPath = join(__dirname, '..', 'output', 'article-script.json');
      await startPreviewServer(scriptPath);
      return;
    }

    // Handle evaluate mode
    if (evaluate) {
      const scriptPath = join(__dirname, '..', 'output', 'article-script.json');
      const result = await evaluateScript(scriptPath);
      printEvaluationResult(result);
      return;
    }

    // Handle check-ass mode
    if (checkAss) {
      const segmentsDir = outputPath || join(__dirname, '..', 'output', 'segments');
      const checkResults = await checkASSFiles(segmentsDir);
      printASSCheckResults(segmentsDir, checkResults);
      return;
    }

    // Phase 1: Segment only (requires input file)
    if (phase === 1) {
      await validatePhase1(inputPath);
      const articleText = await readFile(inputPath, 'utf-8');
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
      await validatePhase2();
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
      await validatePhase3();
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

    // Phase 4: Generate ASS subtitle files (requires phase 3 output)
    if (phase === 4) {
      await validatePhase4();
      logger.info('Phase 4: Generating ASS subtitle files...');
      const phase3Path = join(__dirname, '..', 'output', 'phase-3-result.json');
      const phase3Data = await readFile(phase3Path, 'utf-8');
      const phase3Result = JSON.parse(phase3Data);
      const outputDir = join(__dirname, '..', 'output');

      await runPhase4(phase3Result, outputDir);
      logger.info('Phase 4 complete');
      return;
    }

    // Phase 5: Generate HTML, PNG screenshots and MP4 videos (requires phase 3 and phase 4 output)
    if (phase === 5) {
      await validatePhase5Input();
      logger.info('Phase 5: Generating HTML, screenshots and MP4 videos...');
      const phase3Path = join(__dirname, '..', 'output', 'phase-3-result.json');
      const phase4Path = join(__dirname, '..', 'output', 'phase-4-result.json');
      const articleScriptPath = join(__dirname, '..', 'output', 'article-script.json');
      const phase3Data = await readFile(phase3Path, 'utf-8');
      const phase4Data = await readFile(phase4Path, 'utf-8');
      const articleScriptData = await readFile(articleScriptPath, 'utf-8');
      const phase3Result = JSON.parse(phase3Data);
      const phase4Result = JSON.parse(phase4Data);
      const articleScript = JSON.parse(articleScriptData);
      const outputDir = join(__dirname, '..', 'output');

      await runPhase5(phase3Result, phase4Result, articleScript, outputDir);
      logger.info('Phase 5 complete');
      return;
    }

    // Phase 6: Concatenate all segments into final video (requires phase 5 output)
    if (phase === 6) {
      await validatePhase6Input();
      logger.info('Phase 6: Concatenating segments...');
      if (!outputPath) {
        console.error('Phase 6 requires --output flag');
        process.exit(1);
      }
      const phase5Path = join(__dirname, '..', 'output', 'phase-5-result.json');
      const phase5Data = await readFile(phase5Path, 'utf-8');
      const phase5Result = JSON.parse(phase5Data);

      const allParts: string[] = [];
      if (phase5Result.intro?.videoPath) allParts.push(phase5Result.intro.videoPath);
      for (const seg of phase5Result.segments || []) {
        for (const part of seg.parts || []) {
          if (part.videoPath) allParts.push(part.videoPath);
        }
      }
      if (phase5Result.outro?.videoPath) allParts.push(phase5Result.outro.videoPath);

      if (allParts.length === 0) {
        throw new Error('No video parts to concatenate');
      }

      if (allParts.length === 1) {
        const { rename } = await import('fs/promises');
        await rename(allParts[0], outputPath);
      } else {
        await concatenateSegments(allParts, outputPath);
      }
      logger.info(`Phase 6 complete: ${outputPath}`);
      return;
    }

    // Default: Full pipeline
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
