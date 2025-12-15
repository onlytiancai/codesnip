// 引入 Mocha 和 Chai
import { expect } from 'https://cdn.jsdelivr.net/npm/chai@4.3.7/+esm';

// 引入要测试的模块
import { tokenizePreserve, analyzeText } from '../utils/tokenizer.js';
console.log('tokenizer 模块加载成功', { tokenizePreserve, analyzeText });

// 测试用例
describe('tokenizePreserve', () => {
  it('should tokenize simple text correctly', () => {
    const text = 'Hello world! This is a test.';
    const tokens = tokenizePreserve(text);
    expect(tokens).to.deep.equal(['Hello', 'world', '!', 'This', 'is', 'a', 'test', '.']);
  });

  it('should preserve punctuation', () => {
    const text = 'Hello, world! How are you?';
    const tokens = tokenizePreserve(text);
    expect(tokens).to.deep.equal(['Hello', ',', 'world', '!', 'How', 'are', 'you', '?']);
  });

  it('should handle hyphenated words', () => {
    const text = 'well-known person';
    const tokens = tokenizePreserve(text);
    expect(tokens).to.deep.equal(['well-known', 'person']);
  });

  it('should handle newlines', () => {
    const text = 'Hello\nworld';
    const tokens = tokenizePreserve(text);
    expect(tokens).to.deep.equal(['Hello', '\n', 'world']);
  });

  it('should handle abbreviations', () => {
    const text = 'e.g., i.e., etc.';
    const tokens = tokenizePreserve(text);
    expect(tokens).to.deep.equal(['e.g.', ',', 'i.e.',',', 'etc','.']);
  });

  it('should handle special characters', () => {
    const text = 'Hello\'world it\’s a test';
    const tokens = tokenizePreserve(text);
    expect(tokens).to.deep.equal(['Hello\'world', 'it\’s', 'a', 'test']);
  });
});

describe('analyzeText', () => {
  it('should analyze simple text', async () => {
    const text = 'Hello world!';
    const wordBlocks = [];
    const sentences = { value: [] };
    const fetchIPA = async (word) => `/${word}/`;

    await analyzeText(text, wordBlocks, sentences, fetchIPA);

    expect(wordBlocks).to.have.lengthOf(3);
    expect(wordBlocks[0].word).to.equal('Hello');
    expect(wordBlocks[0].ipa).to.equal('/hello/');
    expect(wordBlocks[1].word).to.equal('world');
    expect(wordBlocks[1].ipa).to.equal('/world/');
    expect(wordBlocks[2].word).to.equal('!');
    expect(wordBlocks[2].ipa).to.equal(null);
    expect(sentences.value).to.have.lengthOf(1);
    expect(sentences.value[0].words).to.have.lengthOf(3);
  });

  it('should handle multiple sentences', async () => {
    const text = 'Hello world! This is a test.';
    const wordBlocks = [];
    const sentences = { value: [] };
    const fetchIPA = async (word) => `/${word}/`;

    await analyzeText(text, wordBlocks, sentences, fetchIPA);

    expect(wordBlocks).to.have.lengthOf(8);
    expect(sentences.value).to.have.lengthOf(2);
    expect(sentences.value[0].words).to.have.lengthOf(3); // Hello, world, !
    expect(sentences.value[1].words).to.have.lengthOf(5); // This, is, a, test, .
  });

  it('should handle hyphenated words', async () => {
    const text = 'well-known person';
    const wordBlocks = [];
    const sentences = { value: [] };
    const fetchIPA = async (word) => `/${word}/`;

    await analyzeText(text, wordBlocks, sentences, fetchIPA);

    expect(wordBlocks).to.have.lengthOf(2);
    expect(wordBlocks[0].word).to.equal('well-known');
    expect(wordBlocks[0].ipa).to.equal('/well/-/known/');
    expect(wordBlocks[1].word).to.equal('person');
    expect(wordBlocks[1].ipa).to.equal('/person/');
  });

  it('should handle empty text', async () => {
    const text = '';
    const wordBlocks = [];
    const sentences = { value: [] };
    const fetchIPA = async (word) => `/${word}/`;

    await analyzeText(text, wordBlocks, sentences, fetchIPA);

    expect(wordBlocks).to.have.lengthOf(0);
    expect(sentences.value).to.have.lengthOf(0);
  });

  it('should handle paragraphs with newlines', async () => {
    const text = 'First paragraph.\n\nSecond paragraph.';
    const wordBlocks = [];
    const sentences = { value: [] };
    const fetchIPA = async (word) => `/${word}/`;

    await analyzeText(text, wordBlocks, sentences, fetchIPA);

    expect(wordBlocks).to.have.lengthOf(6);
    expect(sentences.value).to.have.lengthOf(3); // 两个段落 + 一个换行句子
    expect(sentences.value[0].paragraphIndex).to.equal(0);
    expect(sentences.value[2].paragraphIndex).to.equal(1);
  });
});
