declare module 'subsrt' {
  export function build(captions: unknown[]): string;
  export function parse(content: string): unknown[];
  export function vtt(content: string): string;
  export function srt(content: string): string;
}
