export interface WordWithPhoneme {
  word: string;
  phoneme: string;
}

export interface SentenceAnalysis {
  words: WordWithPhoneme[];
}
