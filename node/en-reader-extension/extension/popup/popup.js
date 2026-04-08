let currentVoice = 'af_bella';
let isSelecting = false;

const voiceSelect = document.getElementById('voice-select');
const selectBtn = document.getElementById('select-btn');
const statusDiv = document.getElementById('status');

voiceSelect.addEventListener('change', (e) => {
  currentVoice = e.target.value;
});

selectBtn.addEventListener('click', async () => {
  if (isSelecting) {
    // Cancel selection
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab) {
      chrome.tabs.sendMessage(tab.id, { action: 'cancelSelection' }).catch(() => {});
    }
    resetButton();
    setStatus('', '');
    return;
  }

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;

  setStatus('loading', 'Loading selection UI...');
  selectBtn.textContent = 'Cancel';
  isSelecting = true;

  chrome.tabs.sendMessage(tab.id, {
    action: 'startSelectionMode',
    voice: currentVoice
  }).then((response) => {
    console.log('startSelectionMode response:', response);
    setStatus('success', 'Click "Select" button on a container');
  }).catch(err => {
    console.error('Failed to start selection mode:', err);
    setStatus('error', `Failed: ${err.message}`);
    resetButton();
  });
});

chrome.runtime.onMessage.addListener((message) => {
  if (message.action === 'selectionCancelled') {
    resetButton();
    setStatus('', '');
  }
});

function setStatus(type, message) {
  statusDiv.className = `status ${type}`;
  statusDiv.textContent = message;
}

function resetButton() {
  selectBtn.textContent = 'Select Content';
  isSelecting = false;
}
