// utils/tokenizer.js - Text tokenization utilities

// Tokenize but preserve original punctuation and special characters
export function tokenizePreserve(textStr) {
  // Match sequences of letters, digits, ASCII apostrophe ' , Unicode right single quote ’,
  // hyphen (to keep n-gram / hyphenated words intact), or any single non-space character.
  // Also match common abbreviations like e.g., i.e., etc. and preserve newlines
  // Modified to include common punctuation (.,!?;:,()) with preceding words
  const re = /(?:[A-Za-z]\.){2,}|[A-Za-z0-9\u2019'-]+[.,!?;:()"'’]*|\n+|[^\s\w]/gu;
  return Array.from(textStr.matchAll(re), m => m[0]);
}

// Analyze text and populate wordBlocks and sentences
export async function analyzeText(text, wordBlocks, sentences, fetchIPA) {
  // Use tokenizer to split text into tokens
  const matches = tokenizePreserve(text || "");

  // Clear existing data
  wordBlocks.splice(0, wordBlocks.length);
  sentences.value = [];

  // Build tokens and sentences simultaneously
  const sentenceSplitRe = /[.,!?，。！；;]/; // include various punctuation
  
  // Current sentence object
  let currentSentence = {
    words: [],
    isNewline: false,
    newline_count: 0,
    sentenceIndex: 0
  };

  for (let w of matches) {
    // Handle newlines
    if (/^\n+$/.test(w)) {
      
      // If current sentence has content, add it to sentences
      if (currentSentence.words.length > 0) {
        sentences.value.push(currentSentence);
      }
      
      // Create newline sentence
      const newlineCount = w.length;
      const newlineSentence = {
        words: [],
        isNewline: true,
        newline_count: newlineCount,
        sentenceIndex: sentences.value.length
      };
      sentences.value.push(newlineSentence);
      
      // Reset current sentence with correct sentenceIndex
      currentSentence = {
        words: [],
        isNewline: false,
        newline_count: 0,
        sentenceIndex: sentences.value.length
      };
      continue;
    }
    
    let wordBlock;
    // Handle hyphenated words
    if (w.includes('-') && /^[A-Za-z0-9\u2019'-]+$/.test(w)) {
      // Split into parts
      const parts = w.split('-');
      const ipas = [];
      
      // Query IPA for each part
      for (let part of parts) {
        const lookup = part.toLowerCase().replace(/[\u2019']/g, "").replace(/[^a-z]/g, "");
        const ipa = lookup ? await fetchIPA(lookup) : null;
        ipas.push(ipa);
      }
      
      // Combine IPA with hyphens
      const combinedIpa = ipas.join('-');
      
      wordBlock = {
        word: w,
        ipa: combinedIpa || null,
        highlight: false,
        sentenceHighlight: false,
        sentenceIndex: currentSentence.sentenceIndex,
        isNewline: false
      };
    } else {
      // Regular words: For IPA lookup, normalize to letters only (remove digits and apostrophes).
      const lookup = w.toLowerCase().replace(/[\u2019']/g, "").replace(/[^a-z]/g, "");
      const ipa = lookup ? await fetchIPA(lookup) : null;
      
      wordBlock = {
        word: w,
        ipa,
        highlight: false,
        sentenceHighlight: false,
        sentenceIndex: currentSentence.sentenceIndex,
        isNewline: false
      };
    }
    
    // Add to wordBlocks and current sentence
    wordBlocks.push(wordBlock);
    currentSentence.words.push(wordBlock);

    // If token matches a split character, add current sentence to sentences
    if (sentenceSplitRe.test(w)) {
      sentences.value.push(currentSentence);
      
      // Reset current sentence with correct sentenceIndex
      currentSentence = {
        words: [],
        isNewline: false,
        newline_count: 0,
        sentenceIndex: sentences.value.length
      };
    }
  }
  
  // If the last sentence has content but wasn't added, add it manually
  if (currentSentence.words.length > 0) {
    sentences.value.push(currentSentence);
  }

  // Reset current sentence index
  return { wordBlocks, sentences };
}
