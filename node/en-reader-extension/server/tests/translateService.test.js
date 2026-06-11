const translateService = require('../services/translateService');

jest.mock('axios');

describe('translateService', () => {
  beforeEach(() => {
    translateService.cache.clear();
    jest.clearAllMocks();
  });

  test('successfully returns translation results', async () => {
    const mockResponse = {
      data: {
        content: [{
          text: JSON.stringify({
            translations: [
              { index: 0, translation: '你好，世界！' },
              { index: 1, translation: '这是一个测试。' }
            ]
          })
        }]
      }
    };

    require('axios').post.mockResolvedValue(mockResponse);

    const result = await translateService.translate(
      '<p>Hello, world!</p>',
      [{ index: 0, tag: 'p', text: 'Hello, world!' }]
    );

    expect(result.translations).toHaveLength(2);
    expect(result.translations[0].translation).toBe('你好，世界！');
  });

  test('caches identical translations', async () => {
    const mockResponse = {
      data: {
        content: [{
          text: JSON.stringify({
            translations: [{ index: 0, translation: '测试' }]
          })
        }]
      }
    };

    require('axios').post.mockResolvedValue(mockResponse);

    const data = { html: '<p>Test</p>', elements: [{ index: 0, tag: 'p', text: 'Test' }] };

    await translateService.translate(data.html, data.elements);
    await translateService.translate(data.html, data.elements);

    expect(require('axios').post).toHaveBeenCalledTimes(1);
  });

  test('handles empty translation from LLM', async () => {
    const mockResponse = {
      data: {
        content: [{
          text: JSON.stringify({
            translations: []
          })
        }]
      }
    };

    require('axios').post.mockResolvedValue(mockResponse);

    const result = await translateService.translate(
      '<p>Hello</p>',
      [{ index: 0, tag: 'p', text: 'Hello' }]
    );

    expect(result.translations).toHaveLength(0);
  });

  test('handles index out of bounds gracefully', async () => {
    const mockResponse = {
      data: {
        content: [{
          text: JSON.stringify({
            translations: [
              { index: 5, translation: 'Out of bounds' }
            ]
          })
        }]
      }
    };

    require('axios').post.mockResolvedValue(mockResponse);

    const result = await translateService.translate(
      '<p>Hello</p>',
      [{ index: 0, tag: 'p', text: 'Hello' }]
    );

    expect(result.translations[0].index).toBe(5);
  });
});
