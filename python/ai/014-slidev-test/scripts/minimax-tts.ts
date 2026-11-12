// MiniMax TTS API 封装
// 文档：https://platform.minimaxi.com/docs/api-reference/speech-t2a-http

export const TTS_ENDPOINT = 'https://api.minimaxi.com/v1/t2a_v2';

export type TTSLanguage = 'zh' | 'en';

export const TTS_VOICES: Record<TTSLanguage, string> = {
  zh: 'female-shaonv',
  en: 'English_PassionateWarrior',
};

export type SynthesizeOptions = {
  text: string;
  language: TTSLanguage;
  voiceId?: string;
  speed?: number;
  vol?: number;
  pitch?: number;
  model?: string;
};

export type SynthesizeResult = {
  buffer: Buffer;
  durationMs: number;
  sampleRate: number;
};

export async function synthesize(opts: SynthesizeOptions): Promise<SynthesizeResult> {
  const apiKey = process.env.MINIMAX_API_KEY;
  if (!apiKey) {
    throw new Error('MINIMAX_API_KEY 环境变量未设置');
  }

  const voiceId = opts.voiceId ?? TTS_VOICES[opts.language];

  const body = {
    model: opts.model ?? 'speech-02-hd',
    text: opts.text,
    stream: false,
    voice_setting: {
      voice_id: voiceId,
      speed: opts.speed ?? 1.0,
      vol: opts.vol ?? 1.0,
      pitch: opts.pitch ?? 0,
    },
    audio_setting: {
      sample_rate: 32000,
      bitrate: 128000,
      format: 'mp3',
    },
  };

  const resp = await fetch(TTS_ENDPOINT, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`TTS HTTP ${resp.status}: ${text}`);
  }

  const json = (await resp.json()) as {
    base_resp?: { status_code: number; status_msg: string };
    data?: { audio: string };
    extra_info?: { audio_length?: number; audio_sample_rate?: number };
  };

  if (json.base_resp?.status_code !== 0) {
    throw new Error(`TTS error ${json.base_resp?.status_code}: ${json.base_resp?.status_msg}`);
  }

  const audioHex = json.data?.audio;
  if (!audioHex) {
    throw new Error('TTS 响应中没有 data.audio');
  }

  const buffer = decodeAudio(audioHex);

  return {
    buffer,
    durationMs: json.extra_info?.audio_length ?? 0,
    sampleRate: json.extra_info?.audio_sample_rate ?? 32000,
  };
}

export function decodeAudio(s: string): Buffer {
  if (/^[0-9a-fA-F]+$/.test(s) && s.length % 2 === 0) {
    return Buffer.from(s, 'hex');
  }
  return Buffer.from(s, 'base64');
}
