import ffmpeg from 'fluent-ffmpeg';
import { mkdir, writeFile } from 'fs/promises';
import { join, dirname } from 'path';
import { VideoSegment, SectionData } from '../types/index.js';
import { logger } from '../utils/logger.js';

interface AssemblyResult {
  segmentPath: string;
  duration: number;
}

/**
 * Assemble a video segment from slide, audio, and subtitles.
 */
export async function assembleSegment(
  slideImage: string,
  audioFile: string,
  subtitleFile: string,
  sectionId: number,
  outputDir: string
): Promise<AssemblyResult> {
  const outputPath = join(outputDir, `segment-${sectionId}.mp4`);

  logger.info(`Assembling video segment ${sectionId}`);

  return new Promise((resolve, reject) => {
    ffmpeg()
      .input(slideImage)
      .inputFormat('image2')
      .inputOptions(['-loop 1'])  // Loop the image for audio duration
      .input(audioFile)
      .inputFormat('mp3')
      .outputOptions([
        '-c:v',
        'libx264',
        '-c:a',
        'aac',
        '-shortest',
        '-pix_fmt',
        'yuv420p',
        '-vf',
        `ass=${subtitleFile}`,
        '-y',
      ])
      .output(outputPath)
      .on('end', async () => {
        const duration = await getVideoDuration(outputPath);
        logger.debug(`Segment ${sectionId} assembled: ${outputPath}, duration: ${duration}s`);
        resolve({ segmentPath: outputPath, duration });
      })
      .on('error', (err) => {
        logger.error(`Failed to assemble segment ${sectionId}:`, err);
        reject(err);
      })
      .run();
  });
}

/**
 * Concatenate multiple video segments into a single video.
 */
export async function concatenateSegments(
  segmentPaths: string[],
  outputPath: string
): Promise<void> {
  if (segmentPaths.length === 0) {
    throw new Error('No segments to concatenate');
  }

  if (segmentPaths.length === 1) {
    // Just rename the single segment
    const { rename } = await import('fs/promises');
    await rename(segmentPaths[0], outputPath);
    return;
  }

  logger.info(`Concatenating ${segmentPaths.length} segments`);

  // Use filter_complex to concatenate videos instead of concat demuxer
  const { resolve: pathResolve } = await import('path');
  const resolvedPaths = segmentPaths.map((p) => pathResolve(p));
  const resolvedOutput = pathResolve(outputPath);

  // Build ffmpeg command with multiple inputs and filter_complex
  const inputs = resolvedPaths.flatMap((p) => ['-i', p]);

  return new Promise((resolve, reject) => {
    const cmd = ffmpeg();

    // Add all input files
    resolvedPaths.forEach((p) => {
      cmd.input(p);
    });

    // Build filter_complex string for concatenation (v=1:a=1 to include audio)
    const filterStrings = resolvedPaths.map(() => '[v]').join('');
    const filterComplex = resolvedPaths.map((_, i) => `[${i}:v][${i}:a]`).join('') + `concat=n=${resolvedPaths.length}:v=1:a=1[outv][outa]`;

    cmd
      .outputOptions([
        '-filter_complex',
        filterComplex,
        '-map',
        '[outv]',
        '-map',
        '[outa]',
        '-y',
      ])
      .output(resolvedOutput)
      .on('end', () => {
        logger.info(`Final video saved to ${outputPath}`);
        resolve();
      })
      .on('error', (err) => {
        logger.error('Failed to concatenate segments:', err);
        reject(err);
      })
      .run();
  });
}

/**
 * Get video duration using ffprobe.
 */
async function getVideoDuration(videoPath: string): Promise<number> {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(videoPath, (err, metadata) => {
      if (err) {
        reject(err);
      } else {
        const duration = metadata.format.duration || 0;
        resolve(duration);
      }
    });
  });
}

/**
 * Assemble complete video from segments.
 */
export async function assembleVideo(
  segments: VideoSegment[],
  outputPath: string
): Promise<void> {
  const segmentPaths: string[] = [];

  for (const segment of segments) {
    const { segmentPath } = await assembleSegment(
      segment.slideImage,
      segment.audioFile,
      segment.subtitleFile,
      segment.section.id,
      join(outputPath, '..', 'segments', `section-${segment.section.id}`)
    );
    segmentPaths.push(segmentPath);
  }

  await concatenateSegments(segmentPaths, outputPath);
}
