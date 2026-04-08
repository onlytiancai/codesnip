const express = require('express');
const translateService = require('../services/translateService');

const router = express.Router();

router.post('/', async (req, res) => {
  try {
    const { html, elements } = req.body;

    if (!html || !elements || !Array.isArray(elements)) {
      return res.status(400).json({ error: 'Invalid request: html and elements are required' });
    }

    const result = await translateService.translate(html, elements);
    res.json(result);
  } catch (error) {
    console.error('Translate error:', error.message);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
