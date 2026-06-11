const { KokoroTTS } = require('kokoro-js');
const path = require('path');

const MODEL_PATH = '/Volumes/data/coscdn/onnx-community/Kokoro-82M-v1.0-ONNX';

let tts = null;

async function getTTS() {
  if (!tts) {
    tts = await KokoroTTS.from_pretrained(MODEL_PATH, {
      dtype: 'q8',
    });
  }
  return tts;
}

class TTSService {
  async synthesize(text, voice = 'af_bella') {
    const kokoro = await getTTS();
    const audio = await kokoro.generate(text, { voice });
    return audio.arrayBuffer();
  }

  getAvailableVoices() {
    return [
      'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore',
      'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
      'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael',
      'am_onyx', 'am_puck', 'am_santa',
      'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
      'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis'
    ];
  }
}

module.exports = new TTSService();
