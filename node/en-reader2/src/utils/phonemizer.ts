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
  
  // 按句子拆分文本
  const sentences = text.split(/[.!?]+/).filter(s => s.trim());
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
