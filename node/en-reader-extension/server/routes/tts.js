const express = require('express');
const ttsService = require('../services/ttsService');

const router = express.Router();

router.post('/', async (req, res) => {
  try {
    const { text, voice } = req.body;

    if (!text) {
      return res.status(400).json({ error: 'Text is required' });
    }

    const audioBuffer = await ttsService.synthesize(text, voice || 'af_bella');

    res.set({
      'Content-Type': 'audio/wav',
      'Content-Length': audioBuffer.length
    });
    res.send(audioBuffer);
  } catch (error) {
    console.error('TTS error:', error.message);
    res.status(500).json({ error: error.message });
  }
});

router.get('/voices', (req, res) => {
  res.json({ voices: ttsService.getAvailableVoices() });
});

module.exports = router;
