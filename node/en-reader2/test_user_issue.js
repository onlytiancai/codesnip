// 测试用户提到的具体问题
import { splitSentencesWithPunctuation } from './src/utils/phonemizer.js';

// 模拟用户的实际输入，确保使用正确的中文引号
const userInput = `"The wildlings are dead."

"Do the dead frighten you?" Ser Waymar Royce asked with just the hint of a smile.`;

console.log('=== Testing User Issue ===');
console.log('Input text:');
console.log(userInput);
console.log('\nInput length:', userInput.length);
console.log('\nCharacter analysis:');
for (let i = 0; i < userInput.length; i++) {
  const char = userInput[i];
  console.log(`Index ${i}: '${char}' (code: ${char.charCodeAt(0)})`);
}

try {
  console.log('\n=== Processing Result ===');
  const result = splitSentencesWithPunctuation(userInput);
  console.log('Number of sentences:', result.length);
  console.log('Sentences:');
  result.forEach((sentence, index) => {
    console.log(`${index + 1}: ${sentence}`);
  });
  
  // 分析问题所在
  if (result.length === 1) {
    console.log('\n=== Problem Analysis ===');
    console.log('The input was not split into multiple sentences.');
    console.log('This suggests the function is not correctly identifying sentence boundaries.');
  }
} catch (error) {
  console.error('Error:', error);
}
