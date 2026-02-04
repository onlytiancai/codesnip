import { splitSentencesWithPunctuation } from './src/utils/phonemizer.js';

// Test cases
const testTexts = [
  "The wildlings are dead.",
  "Hello world. How are you?",
  "Don't split here. It's a contraction.",
  "He said, \"The wildlings are dead.\" Then he left."
];

console.log('Testing sentence segmentation:');
console.log('================================');

testTexts.forEach((text, index) => {
  console.log(`\nTest ${index + 1}: ${text}`);
  console.log('Sentences:');
  const sentences = splitSentencesWithPunctuation(text);
  sentences.forEach((sentence, i) => {
    console.log(`  ${i + 1}: "${sentence.trim()}"`);
  });
});
