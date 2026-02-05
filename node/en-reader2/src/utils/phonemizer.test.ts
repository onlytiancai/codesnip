import { describe, it, expect, vi, beforeEach } from 'vitest';
import { analyzeEnglishText } from './phonemizer';

// 模拟 phonemize 函数
vi.mock('phonemizer', () => ({
  phonemize: vi.fn()
}));

import { phonemize as phonemizeImport } from 'phonemizer';
const mockPhonemize = phonemizeImport as any

describe('analyzeEnglishText', () => {
  beforeEach(() => {
    mockPhonemize.mockClear();
  });

  it('should return empty array for empty input', async () => {
    const result = await analyzeEnglishText('');
    expect(result).toEqual([]);
  });

  it('should return empty array for whitespace only input', async () => {
    const result = await analyzeEnglishText('   ');
    expect(result).toEqual([]);
  });

  it('should handle single word', async () => {
    mockPhonemize.mockResolvedValue(['wɜːrd']);
    
    const result = await analyzeEnglishText('word');
    expect(result).toHaveLength(1);
    expect(result[0]!.words).toHaveLength(1);
    expect(result[0]!.words[0]).toEqual({
      word: 'word',
      phoneme: 'wɜːrd'
    });
  });

  it('should handle multiple words in a sentence', async () => {
    mockPhonemize
      .mockResolvedValueOnce(['hɛˈloʊ'])
      .mockResolvedValueOnce(['wɜːrd']);
    
    const result = await analyzeEnglishText('Hello word');
    expect(result).toHaveLength(1);
    expect(result[0]!.words).toHaveLength(2);
    expect(result[0]!.words[0]).toEqual({
      word: 'Hello',
      phoneme: 'hɛˈloʊ'
    });
    expect(result[0]!.words[1]).toEqual({
      word: 'word',
      phoneme: 'wɜːrd'
    });
  });

  it('should handle multiple sentences', async () => {
    mockPhonemize
      .mockResolvedValueOnce(['hɛˈloʊ'])
      .mockResolvedValueOnce(['wɜːrd'])
      .mockResolvedValueOnce(['wʌts'])
      .mockResolvedValueOnce(['jɔːr'])
      .mockResolvedValueOnce(['neɪm']);
    
    const result = await analyzeEnglishText('Hello word. What\'s your name?');
    expect(result).toHaveLength(2);
    expect(result[0]!.words).toHaveLength(2);
    expect(result[1]!.words).toHaveLength(3);
  });

  it('should handle sentences with quotes', async () => {
    mockPhonemize
      .mockResolvedValueOnce(['hɛˈloʊ'])
      .mockResolvedValueOnce(['saɪd'])
      .mockResolvedValueOnce(['jɔːn']);
    
    const result = await analyzeEnglishText('"Hello," said John.');
    expect(result).toHaveLength(1);
    expect(result[0]!.words).toHaveLength(3);
  });

  it('should handle phonemize error gracefully', async () => {
    mockPhonemize.mockRejectedValue(new Error('Phonemize error'));
    
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    
    const result = await analyzeEnglishText('word');
    expect(result).toHaveLength(1);
    expect(result[0]!.words[0]).toEqual({
      word: 'word',
      phoneme: ''
    });
    
    expect(consoleSpy).toHaveBeenCalledWith('为单词 "word" 生成音标时出错:', expect.any(Error));
    consoleSpy.mockRestore();
  });

  it('should handle phonemize returning empty array', async () => {
    mockPhonemize.mockResolvedValue([]);
    
    const result = await analyzeEnglishText('word');
    expect(result).toHaveLength(1);
    expect(result[0]!.words[0]).toEqual({
      word: 'word',
      phoneme: ''
    });
  });
});
