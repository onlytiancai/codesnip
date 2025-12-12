// main.js - Application entry point
import { createApp, ref, reactive, computed, onMounted, onUnmounted, watchEffect } from './lib/vue/vue.esm-browser.js';
// Import components
import WordBlock from './components/WordBlock.js';
import Sentence from './components/Sentence.js';
import WordTooltip from './components/WordTooltip.js';
// Import utilities
import { loadOfflineIPA, fetchIPA } from './utils/ipa.js';
import { analyzeText } from './utils/tokenizer.js';
import { speakWord, speakSentence, stopSpeech, isSpeechSupported } from './utils/speech.js';
import { lookupWord } from './utils/dictionary.js';
import { translateSentence as translateSentenceApi } from './utils/translation.js';

createApp({
  components: {
    WordBlock,
    Sentence,
    WordTooltip
  },
  setup() {
    // Application state
    const speechSupported = ref(true);
    const text = ref(`Language modeling is a long-standing research topic, dating back to the 1950s with Shannon’s application of information theory to human language.

For example, e.g., i.e., and etc. are common abbreviations.

Low-dimensional embeddings and high-performance computing are important in this field. She carries a book and reads it. The dog is running and barks loudly.

He studied hard and passed the exam.`);

    const isLoadingIPA = ref(false);
    const currentSentenceIndex = ref(0);
    const isSpeaking = ref(false);
    const isLoadingWord = ref(false);
    const selectedWord = ref(null);
    const wordInfo = ref(null);
    
    // Tooltip state
    const tooltipVisible = ref(false);
    const tooltipPosition = ref({ x: 0, y: 0 });
    const tooltipWordIndex = ref(null);
    const tooltipSentenceIndex = ref(null);
    const tooltipWordIdx = ref(null);
    // Translation state
    const currentSentenceTranslation = ref('');
    const isLoadingTranslation = ref(false);
    const translationError = ref(false);
    const isAnalyzing = ref(false);
    const isSettingInitialIndex = ref(false);
    const justAnalyzed = ref(false);
    
    // Settings state
    const isSettingsOpen = ref(false);
    const settings = reactive({
      enableTranslation: true,
      ollamaApiUrl: 'http://localhost:11434/api/generate',
      modelName: 'gemma3:4b',
      translationPrompt: '请将以下英文句子翻译成中文："{sentence}"',
      rate: 1.0,
      pitch: 1.0,
      showIpa: false
    });

    const wordBlocks = reactive([]);
    const sentences = ref([]);
    
    // Computed property for sorted sentence keys
    const sortedKeys = computed(() => {
      return sentences.value.map((_, index) => index);
    });

    // Speech synthesis variables
    let utter = null;
    let stopRequested = false;
    // Translation cancellation variables
    let translationAbortController = null;

    // Check SpeechSynthesis support
    speechSupported.value = isSpeechSupported();

    // Analyze text function
    async function analyze() {
      // Stop any ongoing speech
      stop();
      
      // Reset state variables
      isAnalyzing.value = true;
      isSettingInitialIndex.value = true;
      currentSentenceIndex.value = 0;
      isSpeaking.value = false;
      isLoadingWord.value = false;
      selectedWord.value = null;
      wordInfo.value = null;
      stopRequested = false;
      
      const result = await analyzeText(text.value, wordBlocks, sentences, fetchIPA);
      
      // Translate the first sentence after analysis
      if (settings.enableTranslation && sentences.value.length > 0 && !sentences.value[0].isNewline) {
        const firstSentence = sentences.value[0].words.map(word => word.word).join(' ');
        await translateSentence(firstSentence, {}, settings);
        justAnalyzed.value = true; // 设置标志位，表示刚刚分析完并手动翻译了
      }
      
      // Reset flags
      isAnalyzing.value = false;
      isSettingInitialIndex.value = false;
    }

    // Highlight functions
    function highlightIndex(index) {
      for (let i = 0; i < wordBlocks.length; i++) {
        wordBlocks[i].highlight = (i === index);
      }
    }

    // Hover highlight function
    function hoverIndex(index) {
      for (let i = 0; i < wordBlocks.length; i++) {
        wordBlocks[i].hover = (i === index);
      }
    }

    // Clear hover highlight
    function clearHover() {
      for (let i = 0; i < wordBlocks.length; i++) {
        wordBlocks[i].hover = false;
      }
    }

    // Handle mouse enter word
    function handleWordMouseEnter(index, sentenceIdx, wordIdx) {
      hoverIndex(index);
    }

    // Handle mouse leave word
    function handleWordMouseLeave(index) {
      if (!tooltipVisible.value || tooltipWordIndex.value !== index) {
        // Only clear hover if tooltip is not visible or not for this word
        clearHover();
      }
      // Don't close tooltip when mouse leaves word
    }

    // Close tooltip function
    function closeTooltip() {
      tooltipVisible.value = false;
      tooltipWordIndex.value = null;
      tooltipSentenceIndex.value = null;
      tooltipWordIdx.value = null;
      clearHover();
    }

    // Word information functions
    async function handleGetWordInfo(word, index, sentenceIdx, wordIdx) {
      if (!word || word.trim() === '') {
        return;
      }
      
      // Remove punctuation from word for lookup
      const wordWithoutPunctuation = word.replace(/[.,!?;:()"'’]$/g, '');
      
      isLoadingWord.value = true;
      selectedWord.value = wordWithoutPunctuation;
      wordInfo.value = null;
      
      // Set tooltip position
      const wordElement = document.getElementById(`word-${sentenceIdx}-${wordIdx}`);
      if (wordElement) {
        const rect = wordElement.getBoundingClientRect();
        tooltipPosition.value = {
          x: rect.left + rect.width / 2,
          y: rect.top - 10
        };
      }
      
      try {
        const info = await lookupWord(wordWithoutPunctuation);
        wordInfo.value = info;
      } catch (error) {
        console.error('Failed to get word info:', error);
        wordInfo.value = null;
      } finally {
        isLoadingWord.value = false;
        // Show tooltip regardless of whether word was found or not
        tooltipVisible.value = true;
        tooltipWordIndex.value = index;
        tooltipSentenceIndex.value = sentenceIdx;
        tooltipWordIdx.value = wordIdx;
      }
    }
    
    // Speech functions
    async function speakSentences() {
      await speakSentence(sentences.value, currentSentenceIndex.value, settings.rate, settings.pitch, (sentenceText) => {
        utter = new SpeechSynthesisUtterance(sentenceText);
        utter.lang = "en-US";
        utter.rate = settings.rate;
        utter.pitch = settings.pitch;
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
        await handleGetWordInfo(word.word, index);
      }
      
      speakWord(wordBlocks, index, settings.rate, settings.pitch, (speakText) => {
        utter = new SpeechSynthesisUtterance(speakText);
        utter.lang = "en-US";
        utter.rate = settings.rate;
        utter.pitch = settings.pitch;
        return utter;
      }, (i) => highlightIndex(i), () => {});
    }

    // Translate sentence function using Ollama API
    async function translateSentence(sentence, options = {}, customSettings = null) {
      if (!sentence) {
        currentSentenceTranslation.value = '';
        return;
      }

      // Cancel any ongoing translation
      if (translationAbortController) {
        translationAbortController.abort();
        translationAbortController = null;
      }

      // Create new AbortController for this translation
      translationAbortController = new AbortController();

      isLoadingTranslation.value = true;
      translationError.value = null;
      currentSentenceTranslation.value = '';

      // Use custom settings if provided, otherwise use global settings
      const translationSettings = customSettings || settings;

      try {
        await translateSentenceApi(sentence, {
        onProgress: (progress) => {
          currentSentenceTranslation.value = progress;
        },
        onComplete: () => {
          isLoadingTranslation.value = false;
          translationAbortController = null;
        },
        onError: (error) => {
          console.error('Translation error:', error);
          translationError.value = '翻译失败，请检查Ollama服务是否正常运行';
          currentSentenceTranslation.value = '';
          isLoadingTranslation.value = false;
          translationAbortController = null;
        }
      }, translationSettings, translationAbortController);
      } catch (error) {
        // Ignore AbortError since it's expected when canceling
        if (error.name !== 'AbortError') {
          console.error('Translation error:', error);
          translationError.value = '翻译失败，请检查Ollama服务是否正常运行';
          currentSentenceTranslation.value = '';
        }
        isLoadingTranslation.value = false;
        translationAbortController = null;
      }
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
      // Only handle other keys if speech is supported
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
      // Load settings from localStorage
      loadSettings();
    });

    // Watch for changes in currentSentenceIndex and translate the new sentence
    watchEffect(() => {
      // Skip translation if we're in the middle of analyzing or setting initial index
      if (isAnalyzing.value || isSettingInitialIndex.value) return;
      
      // If we just analyzed and manually translated, skip the first watchEffect run
      if (justAnalyzed.value && currentSentenceIndex.value === 0) {
        justAnalyzed.value = false;
        return;
      }
      
      // Only translate if translation is enabled
      if (settings.enableTranslation && sentences.value.length > 0 && currentSentenceIndex.value >= 0 && currentSentenceIndex.value < sentences.value.length) {
        const currentSentence = sentences.value[currentSentenceIndex.value];
        if (currentSentence && !currentSentence.isNewline) {
          const sentenceText = currentSentence.words.map(word => word.word).join(' ');
          translateSentence(sentenceText, {}, settings);
        }
      }
    });
    
    // Settings functions
    function toggleSettings() {
      isSettingsOpen.value = !isSettingsOpen.value;
    }
    
    function saveSettings() {
      // Save settings to localStorage
      localStorage.setItem('wawaSettings', JSON.stringify(settings));
      // Close settings panel
      isSettingsOpen.value = false;
    }
    
    // Load settings from localStorage
    function loadSettings() {
      const savedSettings = localStorage.getItem('wawaSettings');
      if (savedSettings) {
        const parsedSettings = JSON.parse(savedSettings);
        Object.assign(settings, parsedSettings);
      }
    }

    onUnmounted(() => {
      window.removeEventListener('keydown', handleKeyDown);
    });

    // Return reactive data and methods
    return {
      text,
      isLoadingIPA,
      currentSentenceIndex,
      isSpeaking,
      isLoadingWord,
      selectedWord,
      wordInfo,
      wordBlocks,
      sentences,
      sortedKeys,
      speechSupported,
      // Translation related state
      currentSentenceTranslation,
      isLoadingTranslation,
      translationError,
      // Settings related state and methods
      isSettingsOpen,
      settings,
      toggleSettings,
      saveSettings,
      analyze,
      speakSentences,
      speakPreviousSentence,
      speakNextSentence,
      stop,
      speakWord: handleSpeakWord,
      getWordInfo: handleGetWordInfo,
      translateSentence,
      // Tooltip related state and methods
      tooltipVisible,
      tooltipPosition,
      handleWordMouseEnter,
      handleWordMouseLeave,
      closeTooltip
    };
  }
}).mount('#app');
