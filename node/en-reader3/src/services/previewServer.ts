import http from 'http';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export async function startPreviewServer(scriptPath: string): Promise<void> {
  const previewDir = path.join(__dirname, '..', '..', 'preview');
  const indexPath = path.join(previewDir, 'index.html');

  // Read article-script.json and inject it
  const scriptContent = await fs.readFile(scriptPath, 'utf-8');

  const server = http.createServer(async (req, res) => {
    if (req.url === '/article-script.json') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(scriptContent);
      return;
    }

    if (req.url === '/' || req.url === '/index.html') {
      try {
        const html = await fs.readFile(indexPath, 'utf-8');
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(html);
      } catch {
        res.writeHead(500);
        res.end('Error loading preview page');
      }
      return;
    }

    res.writeHead(404);
    res.end('Not found');
  });

  const PORT = 3001;

  return new Promise((resolve) => {
    server.listen(PORT, () => {
      console.log(`Preview server running at http://localhost:${PORT}`);
      console.log('Press Ctrl+C to stop');
      resolve();
    });
  });
}
