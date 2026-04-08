require('dotenv').config();
const express = require('express');
const translateRouter = require('./routes/translate');
const ttsRouter = require('./routes/tts');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: '10mb' }));

app.use('/api/translate', translateRouter);
app.use('/api/tts', ttsRouter);

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.listen(PORT, () => {
  console.log(`English Reader server running on port ${PORT}`);
});
