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

export enum PartType {
  READING = 'reading',         // Part 1 & 6: English original text
  TRANSLATION = 'translation', // Part 2: Chinese translation narration
  VOCABULARY = 'vocabulary',   // Part 3: Vocabulary explanation
  GRAMMAR = 'grammar',         // Part 4: Grammar explanation
  EXPLANATION = 'explanation', // Part 5: Context/background
}

export interface SegmentPart {
  id: number;
  type: PartType;
  originalText: string;
  script?: string;
  translation?: string;
  vocabulary?: VocabularyItem[];
  grammarPoints?: GrammarPoint[];
  contextExplanation?: string;
}

export interface ArticleScript {
  intro: { title: string; script: string };
  segments: { id: number; originalText: string; parts: SegmentPart[] }[];
  outro: { script: string };
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

// Prompt configuration types
export interface PromptDefinition {
  template: string;
  responseFormat: 'json' | 'text';
  outputFields?: Record<string, string>;
}

export interface PromptsConfig {
  systemPrompt: string;
  temperature: number;
  prompts: {
    intro: PromptDefinition;
    outro: PromptDefinition;
    segment: PromptDefinition;
  };
}

// Segment generation response (single AI call returns all parts)
export interface SegmentGenerationResponse {
  translation: string;
  vocabularyScript: string;
  grammarScript: string;
  contextScript: string;
  vocabulary: VocabularyItem[];
  grammarPoints: GrammarPoint[];
}
