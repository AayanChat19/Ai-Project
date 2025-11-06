// Background service worker for Hallucination Detector

// Open side panel when extension icon is clicked
chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ windowId: tab.windowId });
});

// Listen for messages from the side panel
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'captureSelection') {
    captureSelectedText(message.type)
      .then(result => sendResponse({ success: true, data: result }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true; // Keep message channel open for async response
  }
});

// Capture selected text from the active tab
async function captureSelectedText(type) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  if (!tab || !tab.id) {
    throw new Error('No active tab found');
  }

  // Execute script to get selected text
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      const selection = window.getSelection();
      const selectedText = selection.toString().trim();
      
      if (!selectedText) {
        return { error: 'No text selected' };
      }
      
      return { 
        text: selectedText,
        length: selectedText.length,
        url: window.location.href,
        title: document.title
      };
    }
  });

  if (!results || results.length === 0) {
    throw new Error('Failed to execute script');
  }

  const result = results[0].result;
  
  if (result.error) {
    throw new Error(result.error);
  }

  // Store in chrome.storage.local
  const storageKey = type === 'prompt' ? 'capturedPrompt' : 'capturedResponse';
  await chrome.storage.local.set({
    [storageKey]: {
      text: result.text,
      length: result.length,
      url: result.url,
      title: result.title,
      timestamp: Date.now()
    }
  });

  return result;
}