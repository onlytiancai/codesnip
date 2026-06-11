import { describe, it, expect, vi, beforeEach } from 'vitest';
import { analyzeEnglishText, splitSentencesWithPunctuation } from './phonemizer';

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
      .mockResolvedValueOnce(['jɔːn'])
      .mockResolvedValueOnce(['haʊ'])
      .mockResolvedValueOnce(['ɑː'])
      .mockResolvedValueOnce(['juː']);
    
    const result = await analyzeEnglishText('"Hello," said John. "How are you?"');
    expect(result).toHaveLength(3);
    expect(result[0]!.words).toHaveLength(1);
    expect(result[1]!.words).toHaveLength(2);
    expect(result[2]!.words).toHaveLength(3);
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

describe('splitSentencesWithPunctuation', () => {
  it('should split text with quotes and punctuation correctly', () => {
    const input = `"We should start back," Gared urged as the woods began to grow dark around them.

"The wildlings are dead."

"Do the dead frighten you?" Ser Waymar Royce asked with just the hint of a smile.

Gared did not rise to the bait. He was an old man, past fifty, and he had seen the lordlings come andgo. "Dead is dead," he said. "We have no business with the dead."

"Are they dead?" Royce asked softly. "What proof have we?"

"Will saw them," Gared said. "If he says they are dead, that's proof enough for me."`;
    
    const expected = [
      '"We should start back,"',
      'Gared urged as the woods began to grow dark around them.',
      '"The wildlings are dead."',
      '"Do the dead frighten you?"',
      'Ser Waymar Royce asked with just the hint of a smile.',
      'Gared did not rise to the bait.',
      'He was an old man, past fifty, and he had seen the lordlings come andgo.',
      '"Dead is dead,"',
      'he said.',
      '"We have no business with the dead."',
      '"Are they dead?"',
      'Royce asked softly.',
      '"What proof have we?"',
      '"Will saw them,"',
      'Gared said.',
      '"If he says they are dead, that\'s proof enough for me."'
    ];
    
    const result = splitSentencesWithPunctuation(input);
    expect(result).toEqual(expected);
  });

  it('should handle empty input', () => {
    const result = splitSentencesWithPunctuation('');
    expect(result).toEqual([]);
  });

  it('should handle whitespace only input', () => {
    const result = splitSentencesWithPunctuation('   ');
    expect(result).toEqual([]);
  });

  it('should handle single sentence', () => {
    const input = 'Hello world.';
    const expected = ['Hello world.'];
    const result = splitSentencesWithPunctuation(input);
    expect(result).toEqual(expected);
  });

  it('should handle multiple sentences', () => {
    const input = 'Hello world. How are you? I am fine!';
    const expected = ['Hello world.', 'How are you?', 'I am fine!'];
    const result = splitSentencesWithPunctuation(input);
    expect(result).toEqual(expected);
  });

  it('should handle sentences with quotes', () => {
    const input = '"Hello," said John. "How are you?"';
    const expected = ['"Hello,"', 'said John.', '"How are you?"'];
    const result = splitSentencesWithPunctuation(input);
    expect(result).toEqual(expected);
  });
});
