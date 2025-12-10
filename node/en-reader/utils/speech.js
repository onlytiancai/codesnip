// utils/speech.js - Speech synthesis utilities

// Check if SpeechSynthesis is supported
export function isSpeechSupported() {
  return typeof window !== 'undefined' && 'speechSynthesis' in window && typeof SpeechSynthesisUtterance !== 'undefined';
}

// Speak a single word
export function speakWord(wordBlocks, index, rate, pitch, createUtterance, onStart, onEnd) {
  if (index < 0 || index >= wordBlocks.length) return;

  if (!isSpeechSupported()) return;

  const w = wordBlocks[index].word;
  // Skip punctuation-only tokens
  if (!/[A-Za-z0-9]/.test(w)) return;

  let speakText = w.replace(/[\u2019]/g, "'").replace(/[^A-Za-z0-9'\-]/g, "");
  if (!speakText) speakText = w;

  const utter = createUtterance(speakText);
  utter.lang = "en-US";
  utter.rate = rate;
  utter.pitch = pitch;

  utter.onstart = () => onStart(index);
  utter.onend = () => onEnd();

  window.speechSynthesis.speak(utter);
}

// Speak a sentence
export async function speakSentence(sentences, sentenceIndex, rate, pitch, createUtterance, onStart, onEnd) {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
    onEnd();
    return;
  }

  // Get current sentence from sentences array
  const currentSentence = sentences[sentenceIndex];
  
  // Check if current sentence exists
  if (!currentSentence) {
    onEnd();
    return;
  }

  // Skip reading newline sentences
  if (currentSentence.isNewline) {
    onEnd();
    return;
  }

  // Check if sentence has words
  if (!currentSentence.words || !currentSentence.words.length) {
    onEnd();
    return;
  }

  // Build sentence text by joining words
  const parts = currentSentence.words.map(w => w.word);
  let sentenceText = parts.join(" ");

  // Clean up spacing around punctuation: remove space before punctuation
  sentenceText = sentenceText.replace(/\s+([.,!?;，。！；])/g, "$1");

  const utter = createUtterance(sentenceText);
  utter.lang = "en-US";
  utter.rate = rate;
  utter.pitch = pitch;

  onStart();
  
  await new Promise((resolve) => {
    utter.onstart = () => {
      // Maintain sentence highlight status
    };
    utter.onend = () => {
      onEnd();
      resolve();
    };
    utter.onerror = () => {
      onEnd();
      resolve();
    };
    window.speechSynthesis.speak(utter);
  });
}

// Stop ongoing speech
export function stopSpeech(onStop) {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
  
  window.speechSynthesis.cancel();
  onStop();
}
