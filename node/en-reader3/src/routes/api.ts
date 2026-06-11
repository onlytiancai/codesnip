import { Router, Request, Response, Router as ExpressRouter } from 'express';
import { v4 as uuidv4 } from 'uuid';
import { mkdir } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { segmentArticle } from '../services/segmenter.js';
import { generateScripts } from '../services/scriptGenerator.js';
import { generateTTS } from '../services/tts.js';
import { generateSubtitles } from '../services/subtitleGenerator.js';
import { generateSlide } from '../services/slideGenerator.js';
import { assembleSegment, concatenateSegments } from '../services/videoAssembler.js';
import { JobStatus, SectionData } from '../types/index.js';
import { logger } from '../utils/logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const router: ExpressRouter = Router();

// In-memory job storage (in production, use Redis or a database)
const jobs = new Map<string, JobStatus>();

/**
 * POST /api/generate - Start video generation job
 */
router.post('/generate', async (req: Request, res: Response) => {
  const { articleText, title } = req.body as { articleText: string; title?: string };

  if (!articleText) {
    return res.status(400).json({ error: 'articleText is required' });
  }

  const jobId = uuidv4();
  const outputDir = join(__dirname, '..', '..', '..', 'output', jobId);

  await mkdir(outputDir, { recursive: true });

  jobs.set(jobId, {
    jobId,
    status: 'processing',
    progress: 0,
  });

  // Start generation in background
  generateVideoJob(jobId, articleText, title || 'English Article', outputDir)
    .then((outputPath) => {
      jobs.set(jobId, {
        jobId,
        status: 'completed',
        outputPath,
        progress: 100,
      });
    })
    .catch((error) => {
      logger.error(`Job ${jobId} failed:`, error);
      jobs.set(jobId, {
        jobId,
        status: 'failed',
        error: error.message,
      });
    });

  res.json({ jobId, status: 'processing' });
});

/**
 * GET /api/status/:jobId - Get job status
 */
router.get('/status/:jobId', (req: Request, res: Response) => {
  const job = jobs.get(req.params.jobId);

  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }

  res.json(job);
});

/**
 * GET /api/download/:jobId - Download generated video
 */
router.get('/download/:jobId', async (req: Request, res: Response) => {
  const job = jobs.get(req.params.jobId);

  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }

  if (job.status !== 'completed' || !job.outputPath) {
    return res.status(400).json({ error: 'Job not completed' });
  }

  res.download(job.outputPath);
});

/**
 * Background video generation job.
 */
async function generateVideoJob(
  jobId: string,
  articleText: string,
  title: string,
  outputDir: string
): Promise<string> {
  const segmentsDir = join(outputDir, 'segments');
  await mkdir(segmentsDir, { recursive: true });

  logger.info(`[Job ${jobId}] Starting video generation`);
  logger.info(`[Job ${jobId}] Article length: ${articleText.length} characters`);

  // Step 1: Segment article
  updateProgress(jobId, 5);
  const segments = segmentArticle(articleText);
  logger.info(`[Job ${jobId}] Created ${segments.length} segments`);

  // Step 2: Generate AI scripts
  updateProgress(jobId, 10);
  const scripts = await generateScripts(segments, title);
  updateProgress(jobId, 30);

  // Step 3: Generate TTS audio
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

    const { audioPath, duration } = await generateTTS(
      script.narrationScript,
      sectionDir,
      script.id
    );
    audioData.push({ audioPath, duration });
    scripts[i].audioDuration = duration;

    updateProgress(jobId, 30 + Math.floor((i / scripts.length) * 20));
  }

  // Step 4: Generate slides
  const slidePaths: string[] = [];

  for (let i = 0; i < scripts.length; i++) {
    const script = scripts[i];
    const sectionDir = join(segmentsDir, `section-${script.id}`);
    const slideImage = await generateSlide(script, sectionDir);
    slidePaths.push(slideImage);
    updateProgress(jobId, 50 + Math.floor((i / scripts.length) * 15));
  }

  // Step 5: Generate subtitles
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

  updateProgress(jobId, 65);

  // Step 6: Assemble video segments
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
    updateProgress(jobId, 65 + Math.floor((i / scripts.length) * 20));
  }

  // Step 7: Concatenate segments
  const outputPath = join(outputDir, 'final.mp4');
  await concatenateSegments(segmentPaths, outputPath);
  updateProgress(jobId, 95);

  logger.info(`[Job ${jobId}] Video generation complete: ${outputPath}`);

  return outputPath;
}

function updateProgress(jobId: string, progress: number): void {
  const job = jobs.get(jobId);
  if (job) {
    job.progress = progress;
    jobs.set(jobId, job);
  }
}

export default router;
