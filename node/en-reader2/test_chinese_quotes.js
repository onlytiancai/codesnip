import { splitSentencesWithPunctuation } from './src/utils/phonemizer.js';

// 测试用户提到的中文引号输入
const userInput = `"The wildlings are dead."

"Do the dead frighten you?" Ser Waymar Royce asked with just the hint of a smile.`;

console.log('Testing user input with Chinese quotes:');
console.log('Input:', userInput);

try {
  const result = splitSentencesWithPunctuation(userInput);
  console.log('Result:', result);
  console.log('Number of sentences:', result.length);
  
  // 同时测试英文引号的情况作为对比
  const englishQuotesInput = `"The wildlings are dead."

"Do the dead frighten you?" Ser Waymar Royce asked with just the hint of a smile.`;
  
  console.log('\nTesting with English quotes (for comparison):');
  console.log('Input:', englishQuotesInput);
  
  const englishResult = splitSentencesWithPunctuation(englishQuotesInput);
  console.log('Result:', englishResult);
  console.log('Number of sentences:', englishResult.length);
} catch (error) {
  console.error('Error:', error);
}
