import { phonemize } from 'phonemizer';
import type { SentenceAnalysis, WordWithPhoneme } from '../types';

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
export function splitSentencesWithPunctuation(text: string): string[] {
  // 处理空输入
  if (!text.trim()) return [];
  
  // 替换中文标点符号为英文标点符号
  // 将所有换行符替换为空格
  let normalizedText = text
    .replace(/[，]/g, ',')
    .replace(/[。]/g, '.')
    .replace(/[！]/g, '!')
    .replace(/[？]/g, '?')
    .replace(/[；]/g, ';')
    .replace(/[：]/g, ':')
    .replace(/[“]/g, '"')
    .replace(/[”]/g, '"')
    .replace(/\n/g, ' ')
    .replace(/\s+/g, ' ') // 将多个连续空格替换为单个空格
  
  const sentences: string[] = [];
  let currentSentence = '';
  let inQuotes = false;
  
  for (let i = 0; i < normalizedText.length; i++) {
    const char = normalizedText[i];
    currentSentence += char;
    
    // 处理引号
    if (char === '"') {
      inQuotes = !inQuotes;
      
      // 检查引号内的逗号，如 "Hello," 这种情况
      if (!inQuotes && i > 0 && normalizedText[i - 1] === ',') {
        // 检查下一个字符是否是空格
        const nextChar = i + 1 < normalizedText.length ? normalizedText[i + 1] : '';
        if (nextChar === ' ') {
          // 跳过空格
          while (i + 1 < normalizedText.length && normalizedText[i + 1] === ' ') {
            i++;
          }
          
          const trimmedSentence = currentSentence.trim();
          if (trimmedSentence) {
            sentences.push(trimmedSentence);
          }
          currentSentence = '';
        }
      }
    }
    
    // 检查句子结束符
    if ((char === '.' || char === '!' || char === '?')) {
      // 检查是否在引号内
      if (inQuotes) {
        // 如果在引号内，检查下一个字符是否是引号结束
        if (i + 1 < normalizedText.length && normalizedText[i + 1] === '"') {
          // 消耗引号
          currentSentence += normalizedText[i + 1];
          i++;
          inQuotes = false;
          
          // 检查引号后是否是空格
          const nextChar = i + 1 < normalizedText.length ? normalizedText[i + 1] : '';
          if (nextChar === ' ') {
            // 跳过空格
            while (i + 1 < normalizedText.length && normalizedText[i + 1] === ' ') {
              i++;
            }
            
            const trimmedSentence = currentSentence.trim();
            if (trimmedSentence) {
              sentences.push(trimmedSentence);
            }
            currentSentence = '';
          }
        }
      } else {
        // 不在引号内，检查是否是句子结束
        const nextChar = i + 1 < normalizedText.length ? normalizedText[i + 1] : '';
        if (nextChar === ' ' || nextChar === '' || nextChar === '\n' || nextChar === '\t') {
          // 跳过后续的空格和换行
          while (i + 1 < normalizedText.length && (normalizedText[i + 1] === ' ' || normalizedText[i + 1] === '\n' || normalizedText[i + 1] === '\t')) {
            i++;
          }
          
          const trimmedSentence = currentSentence.trim();
          if (trimmedSentence) {
            sentences.push(trimmedSentence);
          }
          currentSentence = '';
        }
      }
    }
  }
  
  // 处理最后一个句子（如果没有结束符）
  const trimmedSentence = currentSentence.trim();
  if (trimmedSentence) {
    sentences.push(trimmedSentence);
  }
  
  return sentences;
}
