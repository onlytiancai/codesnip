// main.js - Application entry point
import { createApp, ref, reactive, computed, onMounted, onUnmounted, watchEffect, nextTick } from './lib/vue/vue.esm-browser.js';
// Import components
import WordBlock from './components/WordBlock.js';
import Sentence from './components/Sentence.js';
import WordTooltip from './components/WordTooltip.js';
import Sidebar from './components/Sidebar.js';
// Import utilities
import { loadOfflineIPA, fetchIPA } from './utils/ipa.js';
import { analyzeText, setWordBlocksIPA } from './utils/tokenizer.js';
import { speakWord, speakSentence, stopSpeech, isSpeechSupported } from './utils/speech.js';
import { lookupWord } from './utils/dictionary.js';
import { translateSentence as translateSentenceApi } from './utils/translation.js';

createApp({
  components: {
    WordBlock,
    Sentence,
    WordTooltip,
    Sidebar
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
    // Track current selected word index for keyboard navigation
    const currentWordIndex = ref(null);
    // Sidebar state
    const isSidebarOpen = ref(true);
    
    // Load sidebar state from localStorage
    function loadSidebarState() {
      const savedState = localStorage.getItem('isSidebarOpen');
      if (savedState !== null) {
        isSidebarOpen.value = JSON.parse(savedState);
      }
    }
    
    // Save sidebar state to localStorage
    function saveSidebarState() {
      localStorage.setItem('isSidebarOpen', JSON.stringify(isSidebarOpen.value));
    }
    // Sidebar component reference
    const sidebarRef = ref(null);
    
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
      enableParagraphTranslation: true,
      ollamaApiUrl: 'http://localhost:11434/api/generate',
      modelName: 'gemma3:4b',
      translationPrompt: `
You are a professional chinese native translator who needs to fluently translate text into chinese.

## Translation Rules
1. Output only the translated content, without explanations or additional content (such as "Here's the translation:" or "Translation as follows:")
2. The returned translation must maintain exactly the same number of paragraphs and format as the original text
3. If the text contains HTML tags, consider where the tags should be placed in the translation while maintaining fluency
4. For content that should not be translated (such as proper nouns, code, etc.), keep the original text.

Translate to chinese (output translation only):

{sentence}
      `,
      rate: 1.0,
      pitch: 1.0,
      showIpa: false
    });

    const wordBlocks = reactive([]);
    const sentences = ref([]);
    const paragraphs = ref([]);
    
    // Computed property for sorted sentence keys
    const sortedKeys = computed(() => {
      return sentences.value.map((_, index) => index);
    });
    
    // Computed property to build paragraphs from sentences
    const buildParagraphs = computed(() => {
      const paraMap = new Map();
      
      sentences.value.forEach(sentence => {
        const paraIndex = sentence.paragraphIndex;
        if (!paraMap.has(paraIndex)) {
          paraMap.set(paraIndex, {
            sentences: [],
            translation: '',
            isLoading: false,
            error: null
          });
        }
        paraMap.get(paraIndex).sentences.push(sentence);
      });
      
      // Convert map to array and sort by paragraph index
      return Array.from(paraMap.entries())
        .sort((a, b) => a[0] - b[0])
        .map(([index, para]) => {
          return {
            index,
            ...para
          };
        });
    });
    
    // Update paragraphs whenever buildParagraphs changes
    watchEffect(() => {
      paragraphs.value = buildParagraphs.value;
    });

    // Speech synthesis variables
    let utter = null;
    let stopRequested = false;
    // Translation cancellation variables
    let translationAbortController = null;
    let paragraphAbortControllers = new Map();

    // Check SpeechSynthesis support
    speechSupported.value = isSpeechSupported();

    // Sidebar methods
    function toggleSidebar() {
      isSidebarOpen.value = !isSidebarOpen.value;
      saveSidebarState();
    }
    
    function handleSelectText(selectedText) {
      text.value = selectedText;
      analyze();
    }
    
    // Clear text function
    function clearText() {
      text.value = '';
    }
    
    // Paste text function
    async function pasteText() {
      try {
        const clipboardText = await navigator.clipboard.readText();
        text.value = clipboardText;
      } catch (err) {
        console.error('Failed to read clipboard:', err);
      }
    }
    
    // Save current text to recent texts
    async function saveTextToRecent() {
      if (sidebarRef.value && text.value) {
        await sidebarRef.value.saveText(text.value);
        await sidebarRef.value.loadRecentTexts();
      }
    }
    
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
      
      console.log('=== Starting analyze() ===');
      console.log('Text value:', text.value);
      console.log('Initial wordBlocks length:', wordBlocks.length);
      
      // Use the new analyzeText function
      const { wordBlocks: newWordBlocks, sentences: newSentences, paragraphs: newParagraphs } = analyzeText(text.value);
      
      console.log('newWordBlocks length:', newWordBlocks.length);
      console.log('newSentences length:', newSentences.length);
      console.log('newWordBlocks sample:', newWordBlocks.slice(0, 5));
      
      // Set IPA for word blocks
      await setWordBlocksIPA(newWordBlocks, fetchIPA);
      
      // Update reactive data
      wordBlocks.splice(0, wordBlocks.length, ...newWordBlocks);
      sentences.value = newSentences;
      paragraphs.value = newParagraphs;
      
      // Wait for DOM updates and computed properties to recalculate
      await nextTick();
      
      // Translate the first sentence after analysis
      if (settings.enableTranslation && sentences.value.length > 0) {
        // Find the first non-newline sentence by global index
        const firstSentence = sentences.value.find(s => !s.isNewline);
        if (firstSentence) {
          const firstSentenceText = firstSentence.words.map(word => word.word).join(' ');
          await translateSentence(firstSentenceText, {}, settings);
          justAnalyzed.value = true; // 设置标志位，表示刚刚分析完并手动翻译了
        }
      }
      
      // Translate all paragraphs after analysis
      if (settings.enableParagraphTranslation && paragraphs.value.length > 0) {
        for (let i = 0; i < paragraphs.value.length; i++) {
          // Only translate paragraphs that have content
          if (paragraphs.value[i].sentences.some(sent => !sent.isNewline)) {
            await translateParagraph(i, settings);
          }
        }
      }
      
      // Reset flags
      isAnalyzing.value = false;
      isSettingInitialIndex.value = false;
      
      // Save current text to recent texts
      await saveTextToRecent();
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
      
      // Update current word index for keyboard navigation
      currentWordIndex.value = index;
      
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

    // Translate paragraph function using Ollama API
    async function translateParagraph(paragraphIndex, customSettings = null) {
      if (!paragraphs.value[paragraphIndex]) {
        return;
      }

      const paragraph = paragraphs.value[paragraphIndex];
      
      // Get paragraph text by joining all sentences
      const paragraphText = paragraph.sentences
        .filter(sentence => !sentence.isNewline)
        .map(sentence => sentence.words.map(word => word.word).join(' '))
        .join(' ');

      if (!paragraphText) {
        paragraph.translation = '';
        return;
      }

      // Cancel any ongoing translation for this paragraph
      if (paragraphAbortControllers.has(paragraphIndex)) {
        paragraphAbortControllers.get(paragraphIndex).abort();
        paragraphAbortControllers.delete(paragraphIndex);
      }

      // Create new AbortController for this translation
      const abortController = new AbortController();
      paragraphAbortControllers.set(paragraphIndex, abortController);

      paragraph.isLoading = true;
      paragraph.error = null;
      paragraph.translation = '';

      // Use custom settings if provided, otherwise use global settings
      const translationSettings = customSettings || settings;

      try {
        await translateSentenceApi(paragraphText, {
          onProgress: (progress) => {
            paragraph.translation = progress;
          },
          onComplete: () => {
            paragraph.isLoading = false;
            paragraphAbortControllers.delete(paragraphIndex);
          },
          onError: (error) => {
            console.error('Paragraph translation error:', error);
            paragraph.error = '翻译失败，请检查Ollama服务是否正常运行';
            paragraph.translation = '';
            paragraph.isLoading = false;
            paragraphAbortControllers.delete(paragraphIndex);
          }
        }, translationSettings, abortController);
      } catch (error) {
        // Ignore AbortError since it's expected when canceling
        if (error.name !== 'AbortError') {
          console.error('Paragraph translation error:', error);
          paragraph.error = '翻译失败，请检查Ollama服务是否正常运行';
          paragraph.translation = '';
        }
        paragraph.isLoading = false;
        paragraphAbortControllers.delete(paragraphIndex);
      }
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
      
      // Find the previous non-newline sentence by global sentenceIndex
      let prevIndex = currentSentenceIndex.value - 1;
      while (prevIndex >= 0) {
        // Find sentence with this global index
        const sentence = sentences.value.find(s => s.sentenceIndex === prevIndex);
        if (sentence && !sentence.isNewline) {
          break;
        }
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
      
      // Find the next non-newline sentence by global sentenceIndex
      let nextIndex = currentSentenceIndex.value + 1;
      while (true) {
        // Find sentence with this global index
        const sentence = sentences.value.find(s => s.sentenceIndex === nextIndex);
        if (!sentence) {
          // No more sentences
          break;
        }
        if (!sentence.isNewline) {
          // Found a valid sentence
          break;
        }
        nextIndex++;
      }
      
      // Check if we found a valid sentence
      const foundSentence = sentences.value.find(s => s.sentenceIndex === nextIndex);
      if (foundSentence) {
        // Stop current sentence playback if any
        stop();
        currentSentenceIndex.value = nextIndex;
        await speakSentences();
      }
    }

    // Select previous word function
    async function selectPreviousWord() {
      if (!wordBlocks.length) return;
      
      if (currentWordIndex.value === null) {
        // Start from the last word if no word is selected
        currentWordIndex.value = wordBlocks.length - 1;
      } else {
        // Move to previous word
        currentWordIndex.value = (currentWordIndex.value - 1 + wordBlocks.length) % wordBlocks.length;
      }
      
      // Highlight the selected word
      highlightIndex(currentWordIndex.value);
      
      // Speak the selected word
      await handleSpeakWord(currentWordIndex.value);
    }
    
    // Select next word function
    async function selectNextWord() {
      if (!wordBlocks.length) return;
      
      if (currentWordIndex.value === null) {
        // Start from the first word if no word is selected
        currentWordIndex.value = 0;
      } else {
        // Move to next word
        currentWordIndex.value = (currentWordIndex.value + 1) % wordBlocks.length;
      }
      
      // Highlight the selected word
      highlightIndex(currentWordIndex.value);
      
      // Speak the selected word
      await handleSpeakWord(currentWordIndex.value);
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
        case 'ArrowUp':
          event.preventDefault();
          selectPreviousWord();
          break;
        case 'ArrowDown':
          event.preventDefault();
          selectNextWord();
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
      // Load sidebar state from localStorage
      loadSidebarState();
      
      console.log('Before analyze - text value:', text.value);
      console.log('Before analyze - text length:', text.value.length);
      
      // Analyze default text on mount
      await analyze();
      
      console.log('After analyze - wordBlocks length:', wordBlocks.length);
      console.log('After analyze - sentences length:', sentences.value.length);
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
      if (settings.enableTranslation && sentences.value.length > 0) {
        // Find current sentence by global sentenceIndex
        const currentSentence = sentences.value.find(s => s.sentenceIndex === currentSentenceIndex.value);
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
      paragraphs,
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
      // Sidebar related state and methods
      isSidebarOpen,
      sidebarRef,
      toggleSidebar,
      handleSelectText,
      // Other methods
      analyze,
      clearText,
      pasteText,
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
