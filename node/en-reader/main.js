// main.js - Application entry point
import { createApp, ref, reactive, computed, onMounted, onUnmounted } from './lib/vue/vue.esm-browser.js';
// Import components
import WordBlock from './components/WordBlock.js';
import Sentence from './components/Sentence.js';
// Import utilities
import { loadOfflineIPA, fetchIPA } from './utils/ipa.js';
import { analyzeText } from './utils/tokenizer.js';
import { speakWord, speakSentence, stopSpeech, isSpeechSupported } from './utils/speech.js';
import { lookupWord } from './utils/dictionary.js';

createApp({
  components: {
    WordBlock,
    Sentence
  },
  setup() {
    // Application state
    const speechSupported = ref(true);
    const text = ref(`Language modeling is a long-standing research topic, dating back to the 1950s with Shannon’s application of information theory to human language.

For example, e.g., i.e., and etc. are common abbreviations.

Low-dimensional embeddings and high-performance computing are important in this field. She carries a book and reads it. The dog is running and barks loudly.

He studied hard and passed the exam.`);
    const rate = ref(1.0);
    const pitch = ref(1.0);
    const isLoadingIPA = ref(false);
    const showIpa = ref(false);
    const currentSentenceIndex = ref(0);
    const isSpeaking = ref(false);
    const isLoadingWord = ref(false);
    const selectedWord = ref(null);
    const wordInfo = ref(null);

    const wordBlocks = reactive([]);
    const sentences = ref([]);
    
    // Computed property for sorted sentence keys
    const sortedKeys = computed(() => {
      return sentences.value.map((_, index) => index);
    });

    // Speech synthesis variables
    let utter = null;
    let stopRequested = false;

    // Check SpeechSynthesis support
    speechSupported.value = isSpeechSupported();

    // Analyze text function
    async function analyze() {
      // Stop any ongoing speech
      stop();
      
      // Reset state variables
      currentSentenceIndex.value = 0;
      isSpeaking.value = false;
      isLoadingWord.value = false;
      selectedWord.value = null;
      wordInfo.value = null;
      stopRequested = false;
      
      const result = await analyzeText(text.value, wordBlocks, sentences, fetchIPA);
    }

    // Highlight functions
    function highlightIndex(index) {
      for (let i = 0; i < wordBlocks.length; i++) {
        wordBlocks[i].highlight = (i === index);
      }
    }

    // Word information functions
    async function handleGetWordInfo(word) {
      if (!word || word.trim() === '') {
        return;
      }
      
      isLoadingWord.value = true;
      selectedWord.value = word;
      wordInfo.value = null;
      
      try {
        const info = await lookupWord(word);
        wordInfo.value = info;
      } catch (error) {
        console.error('Failed to get word info:', error);
        wordInfo.value = null;
      } finally {
        isLoadingWord.value = false;
      }
    }
    
    // Speech functions
    async function speakSentences() {
      await speakSentence(sentences.value, currentSentenceIndex.value, rate.value, pitch.value, (sentenceText) => {
        utter = new SpeechSynthesisUtterance(sentenceText);
        utter.lang = "en-US";
        utter.rate = rate.value;
        utter.pitch = pitch.value;
        return utter;
      }, () => {
        isSpeaking.value = true;
      }, () => {
        isSpeaking.value = false;
        stopRequested = false;
      });
    }

    // Speak word function
    async function handleSpeakWord(index) {
      // Stop current sentence playback if any
      stop();
      
      // Get the word at the index
      const word = wordBlocks[index];
      if (word && word.word) {
        // Get word info first
        await handleGetWordInfo(word.word);
      }
      
      speakWord(wordBlocks, index, rate.value, pitch.value, (speakText) => {
        utter = new SpeechSynthesisUtterance(speakText);
        utter.lang = "en-US";
        utter.rate = rate.value;
        utter.pitch = pitch.value;
        return utter;
      }, (i) => highlightIndex(i), () => {});
    }

    // Stop speech function
    function stop() {
      stopSpeech(() => {
        stopRequested = true;
        isSpeaking.value = false;
        utter = null;
      });
    }

    // Previous and next sentence functions
    async function speakPreviousSentence() {
      // Ensure wordBlocks is populated
      if (!wordBlocks.length) await analyze();
      
      // Find the previous non-newline sentence
      let prevIndex = currentSentenceIndex.value - 1;
      while (prevIndex >= 0 && sentences.value[prevIndex].isNewline) {
        prevIndex--;
      }
      
      if (prevIndex >= 0) {
        // Stop current sentence playback if any
        stop();
        currentSentenceIndex.value = prevIndex;
        await speakSentences();
      }
    }

    async function speakNextSentence() {
      // Ensure wordBlocks is populated
      if (!wordBlocks.length) await analyze();
      
      // Find the next non-newline sentence
      let nextIndex = currentSentenceIndex.value + 1;
      while (nextIndex < sentences.value.length && sentences.value[nextIndex].isNewline) {
        nextIndex++;
      }
      
      if (nextIndex < sentences.value.length) {
        // Stop current sentence playback if any
        stop();
        currentSentenceIndex.value = nextIndex;
        await speakSentences();
      }
    }

    // Keyboard event handler
    function handleKeyDown(event) {
      // Only handle keys if speech is supported
      if (!speechSupported.value) return;

      // Check if focus is in input/textarea
      const activeElement = document.activeElement;
      const isInInput = activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA');
      
      // Handle different key presses
      switch(event.code) {
        case 'ArrowLeft':
          speakPreviousSentence();
          break;
        case 'ArrowRight':
          speakNextSentence();
          break;
        case 'Space':
          // Allow space in input fields
          if (isInInput) {
            return;
          }
          
          // Prevent default behavior for space to avoid scrolling
          event.preventDefault();
          
          if (isSpeaking.value) {
            stop();
          } else {
            speakSentences();
          }
          break;
      }
    }

    // Setup keyboard event listener
    onMounted(async () => {
      window.addEventListener('keydown', handleKeyDown);
      // Load offline IPA map
      await loadOfflineIPA(isLoadingIPA);
    });

    onUnmounted(() => {
      window.removeEventListener('keydown', handleKeyDown);
    });

    // Return reactive data and methods
    return {
      text,
      rate,
      pitch,
      isLoadingIPA,
      showIpa,
      currentSentenceIndex,
      isSpeaking,
      isLoadingWord,
      selectedWord,
      wordInfo,
      wordBlocks,
      sentences,
      sortedKeys,
      speechSupported,
      analyze,
      speakSentences,
      speakPreviousSentence,
      speakNextSentence,
      stop,
      speakWord: handleSpeakWord,
      getWordInfo: handleGetWordInfo
    };
  }
}).mount('#app');
