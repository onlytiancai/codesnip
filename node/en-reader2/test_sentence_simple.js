/**
 * 按句子拆分文本，考虑引用情况并保留末尾标点
 */
function splitSentencesWithPunctuation(text) {
  const sentences = [];
  let currentSentence = '';
  let inQuotes = 0;
  let quoteChar = '';
  
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    currentSentence += char;
    
    // 处理引号（只处理双引号，单引号通常是撇号）
    if (char === '"') {
      if (inQuotes === 0) {
        // 开始引用
        inQuotes = 1;
        quoteChar = char;
      } else if (inQuotes === 1 && char === quoteChar) {
        // 结束引用
        inQuotes = 0;
        quoteChar = '';
      }
    }
    
    // 句子结束符，且不在引用内
    if ((char === '.' || char === '!' || char === '?') && inQuotes === 0) {
      // 检查下一个字符是否也是标点（如省略号）
      while (i + 1 < text.length && (text[i + 1] === '.' || text[i + 1] === '!' || text[i + 1] === '?')) {
        currentSentence += text[i + 1];
        i++;
      }
      
      // 检查是否是句子结束（后面是空格或文本结束）
      const nextChar = i + 1 < text.length ? text[i + 1] : '';
      if (nextChar === ' ' || nextChar === '' || nextChar === '\n' || nextChar === '\t') {
        sentences.push(currentSentence);
        currentSentence = '';
      }
    }
  }
  
  // 处理最后一个句子（如果没有结束符）
  if (currentSentence.trim()) {
    sentences.push(currentSentence);
  }
  
  return sentences;
}

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
