let currentVoice = 'af_bella';
let isTranslating = false;
let selectionMode = false;
let selectedContainer = null;
let containerButtons = [];

function cleanHTML(element) {
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
}

function extractTranslatableElements(container) {
  const selectors = 'p, h1, h2, h3, h4, h5, h6, li, blockquote';
  const elements = [];

  container.querySelectorAll(selectors).forEach((el, index) => {
    const text = el.innerText.trim();
    if (text.length > 0) {
      elements.push({ index, tag: el.tagName.toLowerCase(), text, element: el });
    }
  });

  return elements;
}

function injectTranslations(elements, translations) {
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
}

function addTTSButtons(elements) {
  elements.forEach(({ index, text }) => {
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    const lastSentence = sentences[sentences.length - 1]?.trim() || text.trim();

    const existingBtn = document.querySelector(`.tts-btn[data-index="${index}"]`);
    if (existingBtn) return;

    const btn = document.createElement('button');
    btn.className = 'tts-btn';
    btn.setAttribute('data-index', index);
    btn.setAttribute('data-text', lastSentence);
    btn.innerHTML = '🔊';
    btn.title = 'Play audio';

    const targetEl = document.querySelector(`.immersive-translate-target[data-index="${index}"]`);
    if (targetEl) {
      targetEl.insertAdjacentElement('afterend', btn);
    }
  });
}

function removeExistingTranslations() {
  document.querySelectorAll('.immersive-translate-target, .tts-btn').forEach(el => el.remove());
}

function removeAllContainerButtons() {
  if (containerButtons && containerButtons.forEach) {
    containerButtons.forEach(btn => {
      // Restore original styles when removing
      const container = btn.parentElement;
      if (container) {
        container.style.outline = '';
        container.style.background = '';
      }
      btn.remove();
    });
  }
  containerButtons = [];
}

function removeSelectionUI() {
  removeAllContainerButtons();
  const translateBtn = document.getElementById('en-reader-translate-btn');
  if (translateBtn) translateBtn.remove();
  const cancelBtn = document.getElementById('en-reader-cancel-btn');
  if (cancelBtn) cancelBtn.remove();
  // Remove highlights
  if (document.querySelectorAll) {
    document.querySelectorAll('.en-reader-selected').forEach(el => {
      el.classList.remove('en-reader-selected');
      el.style.outline = '';
    });
  }
  selectedContainer = null;
}

function createContainerButtons() {
  removeAllContainerButtons();

  // Find main content containers
  const containerSelectors = [
    'div', 'article', 'main', 'section', 'aside',
    'table', 'thead', 'tbody', 'tfoot', 'tr',
    'ul', 'ol', 'li', 'dl', 'dt', 'dd',
    'blockquote', 'figure', 'figcaption',
    'header', 'footer', 'nav', 'address'
  ];

  const containers = document.querySelectorAll(containerSelectors.join(','));

  containers.forEach((container, index) => {
    // Skip tiny containers
    const rect = container.getBoundingClientRect();
    if (rect.width < 50 || rect.height < 30) return;

    // Skip if already has a button or is inside our UI
    if (container.querySelector('.en-reader-container-btn')) return;
    if (container.closest('#en-reader-translate-btn')) return;
    if (container.closest('#en-reader-cancel-btn')) return;

    // Store original styles to restore later
    const originalOutline = container.style.outline;
    const originalBackground = container.style.background;

    const btn = document.createElement('button');
    btn.className = 'en-reader-container-btn';
    btn.textContent = 'Select';
    btn.dataset.index = index;
    btn.style.cssText = `
      position: absolute;
      top: 5px;
      left: 5px;
      z-index: 2147483646;
      padding: 4px 8px;
      background: #007bff;
      color: white;
      border: none;
      border-radius: 4px;
      font-size: 12px;
      cursor: pointer;
      opacity: 0.9;
    `;

    // Add visual indicator to container
    container.style.outline = '2px dashed rgba(0, 123, 255, 0.5)';
    container.style.background = 'rgba(0, 123, 255, 0.05)';

    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      e.preventDefault();

      // Remove previous selections
      if (document.querySelectorAll) {
        document.querySelectorAll('.en-reader-selected').forEach(el => {
          el.classList.remove('en-reader-selected');
          el.style.outline = originalOutline;
          el.style.background = originalBackground;
        });
      }
      removeAllContainerButtons();

      // Highlight selected container
      container.classList.add('en-reader-selected');
      container.style.outline = '3px solid #007bff';
      container.style.background = 'rgba(0, 123, 255, 0.1)';
      selectedContainer = container;

      // Show translate button
      showTranslateButton();
    });

    // Position the container relatively if needed
    const computedStyle = window.getComputedStyle(container);
    if (computedStyle.position === 'static') {
      container.style.position = 'relative';
    }

    container.appendChild(btn);
    containerButtons.push(btn);
  });
}

function showTranslateButton() {
  // Remove existing buttons first
  const existingTranslateBtn = document.getElementById('en-reader-translate-btn');
  if (existingTranslateBtn) existingTranslateBtn.remove();
  const existingCancelBtn = document.getElementById('en-reader-cancel-btn');
  if (existingCancelBtn) existingCancelBtn.remove();

  // Create cancel button
  const cancelBtn = document.createElement('button');
  cancelBtn.id = 'en-reader-cancel-btn';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 140px;
    z-index: 2147483647;
    padding: 12px 24px;
    background: #6c757d;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  `;
  cancelBtn.addEventListener('click', () => {
    exitSelectionMode();
    chrome.runtime.sendMessage({ action: 'selectionCancelled' });
  });
  document.body.appendChild(cancelBtn);

  // Create translate button
  const translateBtn = document.createElement('button');
  translateBtn.id = 'en-reader-translate-btn';
  translateBtn.textContent = 'Translate';
  translateBtn.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 2147483647;
    padding: 12px 32px;
    background: #007bff;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  `;
  translateBtn.addEventListener('click', handleTranslateClick);
  document.body.appendChild(translateBtn);
}

function handleTranslateClick() {
  if (!selectedContainer) {
    alert('Please select a container first by clicking "Select" button on it.');
    return;
  }

  if (isTranslating) return;
  isTranslating = true;

  const translateBtn = document.getElementById('en-reader-translate-btn');
  if (translateBtn) {
    translateBtn.textContent = 'Translating...';
    translateBtn.disabled = true;
  }

  removeExistingTranslations();

  const html = cleanHTML(selectedContainer);
  const elements = extractTranslatableElements(selectedContainer);

  console.log('Translating with elements:', elements);

  if (!elements || elements.length === 0) {
    isTranslating = false;
    if (translateBtn) {
      translateBtn.textContent = 'Translate';
      translateBtn.disabled = false;
    }
    alert('No translatable content found in selected area.');
    return;
  }

  chrome.runtime.sendMessage({
    action: 'translate',
    data: { html, elements }
  }).then(response => {
    console.log('Translate response:', JSON.stringify(response));
    console.log('response.data:', JSON.stringify(response.data));
    console.log('response.data.translations:', JSON.stringify(response.data?.translations));
    if (response && response.success && response.data) {
      let translations = response.data.translations;
      // Handle case where data itself IS the translations object
      if (!translations && response.data.translations === undefined) {
        console.log('data.translations is undefined, checking if data itself has translations');
        translations = response.data;
      }
      console.log('Final translations to use:', JSON.stringify(translations));
      if (translations && Array.isArray(translations)) {
        injectTranslations(elements, translations);
        addTTSButtons(elements);
        if (translateBtn) {
          translateBtn.textContent = 'Done!';
          translateBtn.style.background = '#28a745';
        }
        setTimeout(() => {
          exitSelectionMode();
        }, 1500);
      } else {
        throw new Error('Invalid translation response: translations is not an array');
      }
    } else {
      throw new Error(response?.error || 'Translation failed');
    }
    isTranslating = false;
  }).catch(error => {
    console.error('Translation error:', error);
    if (translateBtn) {
      translateBtn.textContent = 'Translate';
      translateBtn.disabled = false;
    }
    alert('Translation failed: ' + error.message);
    isTranslating = false;
  });
}

function exitSelectionMode() {
  selectionMode = false;
  selectedContainer = null;
  removeSelectionUI();
}

document.addEventListener('click', async (e) => {
  // Handle TTS button click
  if (e.target.classList.contains('tts-btn')) {
    const text = e.target.getAttribute('data-text');
    if (text) {
      try {
        const response = await chrome.runtime.sendMessage({
          action: 'tts',
          data: { text, voice: currentVoice }
        });

        if (response && response.success) {
          const audio = new Audio(response.audioUrl);
          audio.play();
        }
      } catch (error) {
        console.error('TTS error:', error);
      }
    }
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'startSelectionMode') {
    currentVoice = message.voice || 'af_bella';
    selectionMode = true;
    selectedContainer = null;
    removeSelectionUI();
    createContainerButtons();
    showTranslateButton();
    sendResponse({ status: 'selection mode started' });
  } else if (message.action === 'cancelSelection') {
    exitSelectionMode();
    sendResponse({ status: 'cancelled' });
  }
});

window.englishReader = {
  cleanHTML,
  extractTranslatableElements,
  injectTranslations,
  removeExistingTranslations
};
