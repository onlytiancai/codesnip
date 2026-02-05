import { splitSentencesWithPunctuation } from './src/utils/phonemizer.js';

// 测试用户提到的中文引号输入
const userInput = `"The wildlings are dead."

"Do the dead frighten you?" Ser Waymar Royce asked with just the hint of a smile.`;

console.log('Testing user input:');
console.log('Input:', userInput);
console.log('Input length:', userInput.length);
console.log('First 10 characters:', JSON.stringify(userInput.substring(0, 10)));

try {
  const result = splitSentencesWithPunctuation(userInput);
  console.log('Result:', result);
  console.log('Number of sentences:', result.length);
  
  // 分析每个字符
  console.log('\nCharacter analysis:');
  for (let i = 0; i < Math.min(userInput.length, 50); i++) {
    const char = userInput[i];
    console.log(`Index ${i}: '${char}' (code: ${char.charCodeAt(0)})`);
  }
} catch (error) {
  console.error('Error:', error);
}
