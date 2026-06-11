# Kokoro TTS

Kokoro is a frontier TTS model for its size of 82 million parameters (text in/audio out). This JavaScript library allows the model to be run 100% locally in the browser thanks to [🤗 Transformers.js](https://huggingface.co/docs/transformers.js). Try it out using our [online demo](https://huggingface.co/spaces/webml-community/kokoro-webgpu)!

## Usage

First, install the `kokoro-js` library from [NPM](https://npmjs.com/package/kokoro-js) using:

```bash
npm i kokoro-js
```

You can then generate speech as follows:

```js
import { KokoroTTS } from "kokoro-js";

const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
const tts = await KokoroTTS.from_pretrained(model_id, {
  dtype: "q8", // Options: "fp32", "fp16", "q8", "q4", "q4f16"
  device: "wasm", // Options: "wasm", "webgpu" (web) or "cpu" (node). If using "webgpu", we recommend using dtype="fp32".
});

const text = "Life is like a box of chocolates. You never know what you're gonna get.";
const audio = await tts.generate(text, {
  // Use `tts.list_voices()` to list all available voices
  voice: "af_heart",
});
audio.save("audio.wav");
```

Or if you'd prefer to stream the output, you can do that with:

```js
import { KokoroTTS, TextSplitterStream } from "kokoro-js";

const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
const tts = await KokoroTTS.from_pretrained(model_id, {
  dtype: "fp32", // Options: "fp32", "fp16", "q8", "q4", "q4f16"
  // device: "webgpu", // Options: "wasm", "webgpu" (web) or "cpu" (node).
});

// First, set up the stream
const splitter = new TextSplitterStream();
const stream = tts.stream(splitter);
(async () => {
  let i = 0;
  for await (const { text, phonemes, audio } of stream) {
    console.log({ text, phonemes });
    audio.save(`audio-${i++}.wav`);
  }
})();

// Next, add text to the stream. Note that the text can be added at different times.
// For this example, let's pretend we're consuming text from an LLM, one word at a time.
const text = "Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects. It can even run 100% locally in your browser, powered by Transformers.js!";
const tokens = text.match(/\s*\S+/g);
for (const token of tokens) {
  splitter.push(token);
  await new Promise((resolve) => setTimeout(resolve, 10));
}

// Finally, close the stream to signal that no more text will be added.
splitter.close();

// Alternatively, if you'd like to keep the stream open, but flush any remaining text, you can use the `flush` method.
// splitter.flush();
```

## Voices/Samples

> [!TIP]
> You can find samples for each of the voices in the [model card](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX#samples) on Hugging Face.

### American English

| Name         | Traits | Target Quality | Training Duration | Overall Grade |
| ------------ | ------ | -------------- | ----------------- | ------------- |
| **af_heart** | 🚺❤️   |                |                   | **A**         |
| af_alloy     | 🚺     | B              | MM minutes        | C             |
| af_aoede     | 🚺     | B              | H hours           | C+            |
| af_bella     | 🚺🔥   | **A**          | **HH hours**      | **A-**        |
| af_jessica   | 🚺     | C              | MM minutes        | D             |
| af_kore      | 🚺     | B              | H hours           | C+            |
| af_nicole    | 🚺🎧   | B              | **HH hours**      | B-            |
| af_nova      | 🚺     | B              | MM minutes        | C             |
| af_river     | 🚺     | C              | MM minutes        | D             |
| af_sarah     | 🚺     | B              | H hours           | C+            |
| af_sky       | 🚺     | B              | _M minutes_ 🤏    | C-            |
| am_adam      | 🚹     | D              | H hours           | F+            |
| am_echo      | 🚹     | C              | MM minutes        | D             |
| am_eric      | 🚹     | C              | MM minutes        | D             |
| am_fenrir    | 🚹     | B              | H hours           | C+            |
| am_liam      | 🚹     | C              | MM minutes        | D             |
| am_michael   | 🚹     | B              | H hours           | C+            |
| am_onyx      | 🚹     | C              | MM minutes        | D             |
| am_puck      | 🚹     | B              | H hours           | C+            |
| am_santa     | 🚹     | C              | _M minutes_ 🤏    | D-            |

### British English

| Name        | Traits | Target Quality | Training Duration | Overall Grade |
| ----------- | ------ | -------------- | ----------------- | ------------- |
| bf_alice    | 🚺     | C              | MM minutes        | D             |
| bf_emma     | 🚺     | B              | **HH hours**      | B-            |
| bf_isabella | 🚺     | B              | MM minutes        | C             |
| bf_lily     | 🚺     | C              | MM minutes        | D             |
| bm_daniel   | 🚹     | C              | MM minutes        | D             |
| bm_fable    | 🚹     | B              | MM minutes        | C             |
| bm_george   | 🚹     | B              | MM minutes        | C             |
| bm_lewis    | 🚹     | C              | H hours           | D+            |