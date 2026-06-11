const API_BASE = 'http://localhost:3000';

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Background received message:', request.action);

  if (request.action === 'translate') {
    console.log('Sending translate request to server:', request.data);
    fetch(`${API_BASE}/api/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request.data)
    })
      .then(res => {
        console.log('Server response status:', res.status);
        return res.json();
      })
      .then(data => {
        console.log('Server response data:', data);
        sendResponse({ success: true, data });
      })
      .catch(err => {
        console.error('Server error:', err);
        sendResponse({ success: false, error: err.message });
      });
    return true;
  }

  if (request.action === 'tts') {
    fetch(`${API_BASE}/api/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request.data)
    })
      .then(res => res.arrayBuffer())
      .then(buffer => {
        const blob = new Blob([buffer], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        sendResponse({ success: true, audioUrl: url });
      })
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true;
  }

  if (request.action === 'getVoices') {
    fetch(`${API_BASE}/api/tts/voices`)
      .then(res => res.json())
      .then(data => sendResponse({ success: true, data }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true;
  }
});
