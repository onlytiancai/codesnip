import puppeteer, { Browser } from 'puppeteer';
import { createServer, Server } from 'http';
import { readFile, writeFile, mkdir } from 'fs/promises';
import { extname, join } from 'path';
import { logger } from '../utils/logger.js';

export interface RecordingResult {
  outputPath: string;
  duration: number;
}

export interface BrowserRecorderOptions {
  htmlPath: string;
  outputPath: string;
  width: number;
  height: number;
}

interface ServerAddress {
  port: number;
  family?: string;
  address?: string;
}

/**
 * Start a simple HTTP server to serve the HTML file.
 */
function startHttpServer(htmlPath: string): Promise<Server> {
  return new Promise((resolve, reject) => {
    const mimeTypes: Record<string, string> = {
      '.html': 'text/html',
      '.css': 'text/css',
      '.js': 'application/javascript',
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.mp3': 'audio/mpeg',
      '.webm': 'video/webm',
    };

    const server = createServer(async (req, res) => {
      const filePath = htmlPath;
      const ext = extname(filePath).toLowerCase();
      const contentType = mimeTypes[ext] || 'application/octet-stream';

      try {
        const data = await readFile(filePath);
        res.writeHead(200, { 'Content-Type': contentType });
        res.end(data);
      } catch (err) {
        res.writeHead(500);
        res.end('Server error');
      }
    });

    server.on('error', (err) => {
      reject(err);
    });

    server.listen(0, () => {
      resolve(server);
    });
  });
}

/**
 * Capture a screenshot of the HTML slide using Puppeteer.
 * Returns the path to the PNG screenshot.
 */
export async function captureSlideScreenshot(options: BrowserRecorderOptions): Promise<RecordingResult> {
  const { htmlPath, outputPath, width = 1080, height = 1920 } = options;

  let browser: Browser | null = null;
  let server: Server | null = null;

  try {
    // Start HTTP server to serve the HTML file
    server = await startHttpServer(htmlPath);
    const address = server.address() as ServerAddress | null;
    const port = address?.port || 0;
    const serverUrl = `http://localhost:${port}`;

    logger.debug(`HTTP server started at ${serverUrl}`);

    // Launch headless Chrome
    browser = await puppeteer.launch({
      headless: true,
      executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
      ],
    });

    const page = await browser.newPage();

    // Set viewport to 1080x1920 (portrait)
    await page.setViewport({
      width,
      height,
      deviceScaleFactor: 1,
    });

    logger.debug(`Navigating to ${serverUrl}`);
    await page.goto(serverUrl, { waitUntil: 'networkidle0' });

    // Wait for fonts and resources to load
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Get audio duration
    const audioDuration = await page.evaluate(() => {
      const audio = document.getElementById('audio') as HTMLAudioElement | null;
      return audio?.duration || 5;
    }) as number;

    logger.debug(`Audio duration: ${audioDuration}s`);

    // Ensure output directory exists
    const outputDir = join(outputPath, '..');
    await mkdir(outputDir, { recursive: true });

    // Screenshot the HTML slide
    const screenshotPath = outputPath.replace('.png', '.png');
    await page.screenshot({
      path: screenshotPath,
      type: 'png',
      omitBackground: false,
      fullPage: false,
    });

    logger.info(`Screenshot saved to ${screenshotPath}`);

    return {
      outputPath: screenshotPath,
      duration: audioDuration,
    };
  } catch (error) {
    logger.error('Screenshot capture failed:', error);
    throw error;
  } finally {
    if (browser) {
      await browser.close();
    }
    if (server) {
      server.close();
    }
  }
}
