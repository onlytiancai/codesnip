export interface Sentence {
  id: number;
  text: string;
  audio: string | null;
  isRecording: boolean;
  isPlaying: boolean; // 用于回放按钮
  isAiSpeaking: boolean; // 用于AI朗读按钮
}
