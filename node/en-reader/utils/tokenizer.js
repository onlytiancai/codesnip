// utils/tokenizer.js - Text tokenization utilities

// Tokenize but preserve original punctuation and special characters
export function tokenizePreserve(textStr) {
  // Main tokenization regex with clear parts
  // - (?:[A-Za-z]\.){2,}   : Abbreviations (e.g., U.S., Dr.)
  // - [A-Za-z0-9\u2019'-]+ : Words with letters, numbers, apostrophes, and hyphens
  // - [.,!?;:()"'’]+       : Punctuation marks
  // - \n+                  : Newline sequences
  // - [^\s\w]+            : Other special characters not covered by previous patterns
  const tokenRegex = /(?:[A-Za-z]\.){2,}|[A-Za-z0-9\u2019'-]+|[.,!?;:()"'’]+|\n+|[^\s\w]/gu;
  
  // Find all matches and return them as an array
  return Array.from(textStr.matchAll(tokenRegex), match => match[0]);
}

// Analyze text and populate wordBlocks and sentences
export async function analyzeText(text, wordBlocks, sentences, fetchIPA) {
  // Use tokenizer to split text into tokens
  const matches = tokenizePreserve(text || "");

  // Clear existing data
  wordBlocks.splice(0, wordBlocks.length);
  sentences.value = [];

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
        sentences.value.push(currentSentence);
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
      sentences.value.push(newlineSentence);
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
        isNewline: false,
        wordIndex: wordIndex
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
        isNewline: false,
        wordIndex: wordIndex
      };
    }
    
    // Add to wordBlocks and current sentence
    wordBlocks.push(wordBlock);
    currentSentence.words.push(wordBlock);
    wordIndex++;

    // If token matches a split character, add current sentence to sentences
    if (sentenceSplitRe.test(w)) {
      sentences.value.push(currentSentence);
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
    sentences.value.push(currentSentence);
  }

  // Reset current sentence index
  return { wordBlocks, sentences };
}
