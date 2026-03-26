import express from 'express';
import cors from 'cors';
import { config } from './config/index.js';
import apiRoutes from './routes/api.js';
import { logger } from './utils/logger.js';

const app = express();

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// API routes
app.use('/api', apiRoutes);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Error handler
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  logger.error('Server error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

const port = parseInt(config.PORT, 10);

app.listen(port, () => {
  logger.info(`Server listening on http://localhost:${port}`);
});
