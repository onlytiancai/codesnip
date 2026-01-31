export interface Sentence {
  id: number;
  text: string;
  audio: string | null;
  isRecording: boolean;
  isPlaying: boolean;
}
