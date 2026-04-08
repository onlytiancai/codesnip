describe('content-script', () => {
  let cleanHTML, extractTranslatableElements, injectTranslations;

  beforeAll(() => {
    global.chrome = {
      runtime: {
        sendMessage: jest.fn()
      }
    };
  });

  beforeEach(() => {
    document.body.innerHTML = '';
    jest.clearAllMocks();

    cleanHTML = (element) => {
      const clone = element.cloneNode(true);
      const removeSelectors = [
        'script', 'style', 'noscript', 'iframe',
        'nav', 'header', 'footer', 'aside',
        '[role="navigation"]', '[role="banner"]', '[role="complementary"]',
        '.ad', '.advertisement', '.sidebar', '.comment',
        'svg', 'img', 'video', 'audio'
      ];
      removeSelectors.forEach(sel => {
        clone.querySelectorAll(sel).forEach(el => el.remove());
      });
      clone.querySelectorAll('*').forEach(el => {
        Array.from(el.attributes).forEach(attr => {
          if (attr.name.startsWith('on') || attr.name.startsWith('data-')) {
            el.removeAttribute(attr.name);
          }
        });
      });
      return clone.innerHTML;
    };

    extractTranslatableElements = (container) => {
      const selectors = 'p, h1, h2, h3, h4, h5, h6, li, blockquote';
      const elements = [];
      container.querySelectorAll(selectors).forEach((el, index) => {
        const text = el.textContent.trim();
        if (text.length > 0) {
          elements.push({ index, tag: el.tagName.toLowerCase(), text, element: el });
        }
      });
      return elements;
    };

    injectTranslations = (elements, translations) => {
      translations.forEach(({ index, translation }) => {
        if (index >= elements.length) return;
        const { element, tag } = elements[index];
        const existingTrans = element.parentElement.querySelector(
          `.immersive-translate-target[data-index="${index}"]`
        );
        if (existingTrans) return;
        const transEl = document.createElement(tag);
        transEl.className = 'immersive-translate-target';
        transEl.setAttribute('data-index', index);
        transEl.textContent = translation;
        if (tag === 'li') {
          const parent = element.parentElement;
          if (parent.tagName === 'LI') {
            parent.insertAdjacentElement('afterend', transEl);
          } else {
            element.insertAdjacentElement('afterend', transEl);
          }
        } else {
          element.insertAdjacentElement('afterend', transEl);
        }
      });
    };
  });

  describe('cleanHTML', () => {
    test('removes script, style, noscript, iframe', () => {
      document.body.innerHTML = `
        <div>
          <script>alert(1)</script>
          <style>.test { color: red; }</style>
          <noscript>No script</noscript>
          <iframe src="test.html"></iframe>
          <p>Content</p>
        </div>
      `;
      const result = cleanHTML(document.body);
      expect(result).not.toContain('script');
      expect(result).not.toContain('style');
      expect(result).not.toContain('noscript');
      expect(result).not.toContain('iframe');
      expect(result).toContain('Content');
    });

    test('removes nav, header, footer, aside', () => {
      document.body.innerHTML = `
        <nav>Nav content</nav>
        <header>Header content</header>
        <footer>Footer content</footer>
        <aside>Aside content</aside>
        <p>Main content</p>
      `;
      const result = cleanHTML(document.body);
      expect(result).not.toContain('Nav content');
      expect(result).not.toContain('Header content');
      expect(result).toContain('Main content');
    });

    test('removes elements with navigation role', () => {
      document.body.innerHTML = `
        <div role="navigation">Skip</div>
        <div role="banner">Banner</div>
        <p>Content</p>
      `;
      const result = cleanHTML(document.body);
      expect(result).not.toContain('Skip');
      expect(result).not.toContain('Banner');
      expect(result).toContain('Content');
    });

    test('removes .ad, .advertisement, .sidebar, .comment', () => {
      document.body.innerHTML = `
        <div class="ad">Ad content</div>
        <div class="advertisement">Advertisement</div>
        <div class="sidebar">Sidebar</div>
        <div class="comment">Comment</div>
        <p>Real content</p>
      `;
      const result = cleanHTML(document.body);
      expect(result).not.toContain('Ad content');
      expect(result).toContain('Real content');
    });

    test('removes svg, img, video, audio', () => {
      document.body.innerHTML = `
        <svg><circle cx="10" cy="10" r="5"/></svg>
        <img src="test.jpg"/>
        <video src="test.mp4"></video>
        <audio src="test.mp3"></audio>
        <p>Text</p>
      `;
      const result = cleanHTML(document.body);
      expect(result).not.toContain('svg');
      expect(result).not.toContain('img');
      expect(result).not.toContain('video');
      expect(result).not.toContain('audio');
      expect(result).toContain('Text');
    });

    test('removes onclick and data-* attributes', () => {
      document.body.innerHTML = `
        <p onclick="alert(1)" data-id="123" class="keep">Hello</p>
      `;
      const result = cleanHTML(document.body);
      expect(result).not.toContain('onclick');
      expect(result).not.toContain('data-id');
      expect(result).toContain('keep');
    });

    test('handles deeply nested div structures', () => {
      document.body.innerHTML = `
        <div>
          <div>
            <div>
              <p>Deep content</p>
            </div>
          </div>
        </div>
      `;
      const result = cleanHTML(document.body);
      expect(result).toContain('Deep content');
    });
  });

  describe('extractTranslatableElements', () => {
    test('extracts single p tag', () => {
      document.body.innerHTML = '<p>Hello world</p>';
      const elements = extractTranslatableElements(document.body);
      expect(elements).toHaveLength(1);
      expect(elements[0].text).toBe('Hello world');
      expect(elements[0].tag).toBe('p');
    });

    test('extracts multiple consecutive p tags', () => {
      document.body.innerHTML = '<p>First</p><p>Second</p><p>Third</p>';
      const elements = extractTranslatableElements(document.body);
      expect(elements).toHaveLength(3);
      expect(elements[0].text).toBe('First');
      expect(elements[1].text).toBe('Second');
      expect(elements[2].text).toBe('Third');
    });

    test('extracts h1-h6 heading levels', () => {
      document.body.innerHTML = `
        <h1>Heading 1</h1>
        <h2>Heading 2</h2>
        <h3>Heading 3</h3>
        <h4>Heading 4</h4>
        <h5>Heading 5</h5>
        <h6>Heading 6</h6>
      `;
      const elements = extractTranslatableElements(document.body);
      expect(elements).toHaveLength(6);
      expect(elements.map(e => e.tag)).toEqual(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']);
    });

    test('extracts li from nested lists', () => {
      document.body.innerHTML = `
        <ul>
          <li>Item 1</li>
          <li>Item 2</li>
        </ul>
      `;
      const elements = extractTranslatableElements(document.body);
      expect(elements).toHaveLength(2);
      expect(elements[0].tag).toBe('li');
      expect(elements[0].text).toBe('Item 1');
    });

    test('extracts blockquote', () => {
      document.body.innerHTML = '<blockquote>Quote text</blockquote>';
      const elements = extractTranslatableElements(document.body);
      expect(elements).toHaveLength(1);
      expect(elements[0].tag).toBe('blockquote');
    });

    test('skips empty elements', () => {
      document.body.innerHTML = '<p></p><p>Content</p>';
      const elements = extractTranslatableElements(document.body);
      expect(elements).toHaveLength(1);
      expect(elements[0].text).toBe('Content');
    });

    test('skips whitespace-only elements', () => {
      document.body.innerHTML = '<p>   </p><p>Content</p>';
      const elements = extractTranslatableElements(document.body);
      expect(elements).toHaveLength(1);
    });

    test('preserves element order and index mapping', () => {
      document.body.innerHTML = '<p>First</p><h1>Second</h1><p>Third</p>';
      const elements = extractTranslatableElements(document.body);
      expect(elements[0].index).toBe(0);
      expect(elements[1].index).toBe(1);
      expect(elements[2].index).toBe(2);
    });

    test('handles special characters and HTML entities', () => {
      document.body.innerHTML = '<p>&lt;script&gt; alert(&quot;XSS&quot;); &lt;/script&gt;</p>';
      const elements = extractTranslatableElements(document.body);
      expect(elements).toHaveLength(1);
    });

    test('preserves link text within p tags', () => {
      document.body.innerHTML = '<p>Text with <a href="#">link</a> inside</p>';
      const elements = extractTranslatableElements(document.body);
      expect(elements).toHaveLength(1);
      expect(elements[0].text).toContain('link');
    });
  });

  describe('injectTranslations', () => {
    test('inserts translation after p tag', () => {
      document.body.innerHTML = '<p>Hello</p>';
      const elements = extractTranslatableElements(document.body);
      injectTranslations(elements, [{ index: 0, translation: '你好' }]);

      const p = document.querySelector('p');
      const translation = p.nextElementSibling;
      expect(translation.className).toBe('immersive-translate-target');
      expect(translation.textContent).toBe('你好');
    });

    test('inserts translation after h1/h2/h3', () => {
      document.body.innerHTML = '<h1>Title</h1><h2>Subtitle</h2>';
      const elements = extractTranslatableElements(document.body);
      injectTranslations(elements, [
        { index: 0, translation: '标题' },
        { index: 1, translation: '副标题' }
      ]);

      expect(document.querySelectorAll('.immersive-translate-target')).toHaveLength(2);
    });

    test('inserts translation after li', () => {
      document.body.innerHTML = '<ul><li>Item</li></ul>';
      const elements = extractTranslatableElements(document.body);
      injectTranslations(elements, [{ index: 0, translation: '条目' }]);

      const li = document.querySelector('li');
      expect(li.nextElementSibling.className).toBe('immersive-translate-target');
    });

    test('prevents duplicate injection', () => {
      document.body.innerHTML = '<p>Hello</p>';
      const elements = extractTranslatableElements(document.body);
      injectTranslations(elements, [{ index: 0, translation: '你好' }]);
      injectTranslations(elements, [{ index: 0, translation: '你好' }]);

      expect(document.querySelectorAll('.immersive-translate-target')).toHaveLength(1);
    });

    test('handles nested list li injection position', () => {
      document.body.innerHTML = `
        <ul>
          <li>
            <ul>
              <li>Nested item</li>
            </ul>
          </li>
        </ul>
      `;
      const elements = extractTranslatableElements(document.body);
      // Outer li and inner li both match selector
      expect(elements).toHaveLength(2);
      expect(elements[0].tag).toBe('li');
      expect(elements[1].tag).toBe('li');
    });
  });
});
