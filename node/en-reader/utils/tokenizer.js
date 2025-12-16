// utils/tokenizer.js - Text tokenization utilities

// Tokenize but preserve original punctuation and special characters
export function tokenizePreserve(textStr) {
  // Main tokenization regex with clear parts
  // - (?:[A-Za-z]\.){2,}                 : Abbreviations (e.g., U.S., Dr.)
  // - [A-Za-z0-9]+(?:[\u2019'-./][A-Za-z0-9.]+)* : Words with apostrophes, hyphens, dots, and slashes (e.g., GPT-5.2, v2.0.1, example.com, docs/v1.2.3, real-world)
  // - [.,!?;:()"'’]+                      : Punctuation marks
  // - \n+                                 : Newline sequences
  // - [^\s\w]+                           : Other special characters not covered by previous patterns
  const tokenRegex = /(?:[A-Za-z]\.){2,}|[A-Za-z0-9]+(?:[\u2019'-./][A-Za-z0-9.]+)*|[.,!?;:()"'’]+|\n+|[^\s\w]/gu;
  
  // Find all matches and return them as an array
  return Array.from(textStr.matchAll(tokenRegex), match => match[0]);
}

// Analyze text and return wordBlocks, sentences and paragraphs
export function analyzeText(text) {
  // Use tokenizer to split text into tokens
  const matches = tokenizePreserve(text || "");

  // Initialize data structures
  const wordBlocks = [];
  const sentences = [];

  // Build tokens and sentences simultaneously
  const sentenceSplitRe = /[.!?！;]/; // only use sentence-ending punctuation (excluding commas)
  
  // Current sentence object
  let currentSentence = {
    words: [],
    isNewline: false,
    newline_count: 0,
    sentenceIndex: 0,
    paragraphIndex: 0
  };

  let wordIndex = 0;
  let paragraphIndex = 0;
  let globalSentenceIndex = 0; // Track global sentence index across all paragraphs

  for (let w of matches) {
    // Handle newlines
    if (/^\n+$/.test(w)) {
      
      // If current sentence has content, add it to sentences
      if (currentSentence.words.length > 0) {
        sentences.push(currentSentence);
      }
      
      // Create newline sentence
      const newlineCount = w.length;
      const newlineSentence = {
        words: [],
        isNewline: true,
        newline_count: newlineCount,
        sentenceIndex: globalSentenceIndex,
        paragraphIndex: paragraphIndex
      };
      sentences.push(newlineSentence);
      globalSentenceIndex++;
      
      // Check if this is a paragraph break (2 or more newlines)
      if (newlineCount >= 2) {
        paragraphIndex++;
      }
      
      // Reset current sentence with correct sentenceIndex and paragraphIndex
      currentSentence = {
        words: [],
        isNewline: false,
        newline_count: 0,
        sentenceIndex: globalSentenceIndex,
        paragraphIndex: paragraphIndex
      };
      continue;
    }
    
    // Create word block without IPA
    const wordBlock = {
      word: w,
      ipa: null, // Will be set later
      highlight: false,
      sentenceHighlight: false,
      sentenceIndex: currentSentence.sentenceIndex,
      isNewline: false,
      wordIndex: wordIndex
    };
    
    // Add to wordBlocks and current sentence
    wordBlocks.push(wordBlock);
    currentSentence.words.push(wordBlock);
    wordIndex++;

    // If token is a sentence-ending punctuation mark (not part of a word like version number),
    // add current sentence to sentences
    if (sentenceSplitRe.test(w) && /^[.!?！;]+$/.test(w)) {
      sentences.push(currentSentence);
      globalSentenceIndex++;
      
      // Reset current sentence with correct sentenceIndex and paragraphIndex
      currentSentence = {
        words: [],
        isNewline: false,
        newline_count: 0,
        sentenceIndex: globalSentenceIndex,
        paragraphIndex: paragraphIndex
      };
    }
  }
  
  // If the last sentence has content but wasn't added, add it manually
  if (currentSentence.words.length > 0) {
    sentences.push(currentSentence);
  }

  // Build paragraphs from sentences
  const paragraphs = buildParagraphs(sentences);

  return { wordBlocks, sentences, paragraphs };
}

// Helper function to build paragraphs from sentences
function buildParagraphs(sentences) {
  const paraMap = new Map();
  
  sentences.forEach(sentence => {
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
}

// Set IPA for word blocks
export async function setWordBlocksIPA(wordBlocks, fetchIPA) {
  for (let wordBlock of wordBlocks) {
    const { word } = wordBlock;
    
    // Skip punctuation and non-alphanumeric tokens
    if (/^[^a-zA-Z0-9]+$/.test(word)) {
      wordBlock.ipa = null;
      continue;
    }
    
    // Handle hyphenated words
    if (word.includes('-') && /^[A-Za-z0-9\u2019'-]+$/.test(word)) {
      // Split into parts
      const parts = word.split('-');
      const ipas = [];
      
      // Query IPA for each part
      for (let part of parts) {
        const lookup = part.toLowerCase().replace(/[\u2019']/g, "").replace(/[^a-z]/g, "");
        const ipa = lookup ? await fetchIPA(lookup) : null;
        ipas.push(ipa);
      }
      
      // Combine IPA with hyphens
      wordBlock.ipa = ipas.join('-');
    } else {
      // Regular words: For IPA lookup, normalize to letters only (remove digits and apostrophes)
      const lookup = word.toLowerCase().replace(/[\u2019']/g, "").replace(/[^a-z]/g, "");
      wordBlock.ipa = lookup ? await fetchIPA(lookup) : null;
    }
  }
  
  return wordBlocks;
}
