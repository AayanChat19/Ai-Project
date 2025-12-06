// Background service worker for Hallucination Detector Extension

console.log('Hallucination Detector background service worker loaded');

// Listen for messages from the side panel
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'captureSelection') {
    handleCaptureSelection(request.type)
      .then(sendResponse)
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true;
  }
});

// Handle text selection capture
async function handleCaptureSelection(type) {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab) {
      throw new Error('No active tab found');
    }

    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        const selection = window.getSelection();
        return {
          text: selection.toString().trim(),
          url: window.location.href,
          title: document.title
        };
      }
    });

    const result = results[0].result;

    if (!result.text) {
      throw new Error('No text selected. Please highlight some text on the page.');
    }

    const capturedData = {
      text: result.text,
      url: result.url,
      pageTitle: result.title,
      timestamp: Date.now(),
      length: result.text.length
    };

    const storageKey = type === 'prompt' ? 'capturedPrompt' : 'capturedResponse';
    await chrome.storage.local.set({ [storageKey]: capturedData });

    console.log(`Captured ${type}:`, capturedData);

    return {
      success: true,
      data: capturedData
    };

  } catch (error) {
    console.error(`Error capturing ${type}:`, error);
    return {
      success: false,
      error: error.message
    };
  }
}

// Handle extension icon click - open side panel
chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ windowId: tab.windowId });
});

chrome.runtime.onInstalled.addListener(() => {
  console.log('Hallucination Detector extension installed');
});
