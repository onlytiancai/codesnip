import { phonemize } from 'phonemizer';

export interface WordWithPhoneme {
  word: string;
  phoneme: string;
}

export interface SentenceAnalysis {
  words: WordWithPhoneme[];
}

/**
 * 分析英文文本，按句子拆分并为每个单词生成音标
 */
export async function analyzeEnglishText(text: string): Promise<SentenceAnalysis[]> {
  if (!text.trim()) return [];
  
  // 按句子拆分文本，考虑引用和保留标点
  const sentences = splitSentencesWithPunctuation(text);
  const results: SentenceAnalysis[] = [];
  
  for (const sentence of sentences) {
    const words = sentence.trim().split(/\s+/).filter(w => w);
    if (words.length === 0) continue;
    
    // 为每个单词生成音标
    const wordsWithPhonemes: WordWithPhoneme[] = [];
    for (const word of words) {
      try {
        const phoneme = await phonemize(word);
        wordsWithPhonemes.push({
          word,
          phoneme: phoneme[0] || ''
        });
      } catch (error) {
        console.error(`为单词 "${word}" 生成音标时出错:`, error);
        wordsWithPhonemes.push({
          word,
          phoneme: ''
        });
      }
    }
    
    results.push({ words: wordsWithPhonemes });
  }
  
  return results;
}

/**
 * 按句子拆分文本，考虑引用情况并保留末尾标点
 */
function splitSentencesWithPunctuation(text: string): string[] {
  const sentences: string[] = [];
  let currentSentence = '';
  let inQuotes = 0;
  let quoteChar = '';
  
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    currentSentence += char;
    
    // 处理引号
    if (char === '"' || char === "'") {
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
      
      sentences.push(currentSentence);
      currentSentence = '';
    }
  }
  
  // 处理最后一个句子（如果没有结束符）
  if (currentSentence.trim()) {
    sentences.push(currentSentence);
  }
  
  return sentences;
}
