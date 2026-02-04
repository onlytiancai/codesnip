/**
 * 按句子拆分文本，考虑引用情况并保留末尾标点
 */
function splitSentencesWithPunctuation(text) {
  const sentences = [];
  let currentSentence = '';
  let inQuotes = 0;
  let quoteChar = '';
  let lastPunctuationIndex = -1;
  
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
        
        // 检查引号前是否有标点，如果有，可能是句子结束
        if (lastPunctuationIndex !== -1) {
          // 检查下一个字符是否是空格或文本结束
          const nextChar = i + 1 < text.length ? text[i + 1] : '';
          if (nextChar === ' ' || nextChar === '' || nextChar === '\n' || nextChar === '\t') {
            sentences.push(currentSentence);
            currentSentence = '';
            lastPunctuationIndex = -1;
          }
        }
      }
    }
    
    // 记录标点位置
    if ((char === '.' || char === '!' || char === '?')) {
      lastPunctuationIndex = i;
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
        lastPunctuationIndex = -1;
      }
    }
  }
  
  // 处理最后一个句子（如果没有结束符）
  if (currentSentence.trim()) {
    sentences.push(currentSentence);
  }
  
  return sentences;
}

// Test the user's example
const userExample = "\"We have no business with the dead.\" \"Are they dead?\" Royce asked softly.";

console.log('Testing user example:');
console.log('================================');
console.log(`Input: ${userExample}`);
console.log('Sentences:');

const sentences = splitSentencesWithPunctuation(userExample);
sentences.forEach((sentence, i) => {
  console.log(`  ${i + 1}: "${sentence.trim()}"`);
});

// Test the original wildlings example
const wildlingsExample = "The wildlings are dead.";
console.log('\nTesting wildlings example:');
console.log(`Input: ${wildlingsExample}`);
const wildlingsSentences = splitSentencesWithPunctuation(wildlingsExample);
wildlingsSentences.forEach((sentence, i) => {
  console.log(`  ${i + 1}: "${sentence.trim()}"`);
});
