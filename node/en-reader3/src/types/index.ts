export interface VocabularyItem {
  word: string;
  phonetic: string;
  partOfSpeech: string;
  definition: string;
  example: string;
}

export interface PhraseItem {
  phrase: string;
  meaning: string;
  example: string;
}

export interface GrammarPoint {
  rule: string;
  explanation: string;
  example: string;
}

export interface ArticleSection {
  id: number;
  originalText: string;
  summary: string;
  vocabulary: VocabularyItem[];
  phrases: PhraseItem[];
  grammarPoints: GrammarPoint[];
  contextExplanation: string;
  narrationScript: string;
  audioDuration?: number;
}

export interface SectionData {
  id: number;
  originalText: string;
  summary: string;
  vocabulary: VocabularyItem[];
  grammarPoints: GrammarPoint[];
  contextExplanation: string;
  narrationScript: string;
  audioDuration?: number;
}

export interface VideoSegment {
  section: SectionData;
  slideImage: string;
  audioFile: string;
  subtitleFile: string;
}

export interface VideoOptions {
  outputPath: string;
  title?: string;
  jobId?: string;
}

export interface JobStatus {
  jobId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  outputPath?: string;
  error?: string;
  progress?: number;
}
