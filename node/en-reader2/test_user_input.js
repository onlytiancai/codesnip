import { splitSentencesWithPunctuation } from './src/utils/phonemizer.js';

// 用户提供的输入
const userInput = '”We should start back,” Gared urged as the woods began to grow dark around them.“The wildlings are dead.”“Do the dead frighten you?” ';

console.log('Testing user input:');
console.log('================================');
console.log('Input:', userInput);
console.log('\nSentences:');

const sentences = splitSentencesWithPunctuation(userInput);
sentences.forEach((sentence, i) => {
  console.log(`  ${i + 1}: "${sentence.trim()}"`);
});

console.log('\nTotal sentences:', sentences.length);
