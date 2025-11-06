// Side panel JavaScript for Hallucination Detector

// DOM elements
const capturePromptBtn = document.getElementById('capturePromptBtn');
const captureResponseBtn = document.getElementById('captureResponseBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const promptPreview = document.getElementById('promptPreview');
const responsePreview = document.getElementById('responsePreview');
const promptStatus = document.getElementById('promptStatus');
const responseStatus = document.getElementById('responseStatus');
const promptMeta = document.getElementById('promptMeta');
const responseMeta = document.getElementById('responseMeta');
const promptError = document.getElementById('promptError');
const responseError = document.getElementById('responseError');
const resultCard = document.getElementById('resultCard');
const scoreValue = document.getElementById('scoreValue');
const scoreDescription = document.getElementById('scoreDescription');

// State
let capturedPrompt = null;
let capturedResponse = null;

// Initialize: Load stored data
async function initialize() {
  const data = await chrome.storage.local.get(['capturedPrompt', 'capturedResponse']);
  
  if (data.capturedPrompt) {
    capturedPrompt = data.capturedPrompt;
    updatePromptUI(capturedPrompt);
  }
  
  if (data.capturedResponse) {
    capturedResponse = data.capturedResponse;
    updateResponseUI(capturedResponse);
  }
  
  updateAnalyzeButton();
}

// Capture prompt
capturePromptBtn.addEventListener('click', async () => {
  capturePromptBtn.disabled = true;
  capturePromptBtn.innerHTML = '<span class="loading"></span>';
  promptError.classList.remove('show');
  
  try {
    const response = await chrome.runtime.sendMessage({
      action: 'captureSelection',
      type: 'prompt'
    });
    
    if (response.success) {
      capturedPrompt = response.data;
      updatePromptUI(capturedPrompt);
      updateAnalyzeButton();
    } else {
      throw new Error(response.error);
    }
  } catch (error) {
    promptError.textContent = `Error: ${error.message}`;
    promptError.classList.add('show');
  } finally {
    capturePromptBtn.disabled = false;
    capturePromptBtn.textContent = 'Capture Selected Text as Prompt';
  }
});

// Capture response
captureResponseBtn.addEventListener('click', async () => {
  captureResponseBtn.disabled = true;
  captureResponseBtn.innerHTML = '<span class="loading"></span>';
  responseError.classList.remove('show');
  
  try {
    const response = await chrome.runtime.sendMessage({
      action: 'captureSelection',
      type: 'response'
    });
    
    if (response.success) {
      capturedResponse = response.data;
      updateResponseUI(capturedResponse);
      updateAnalyzeButton();
    } else {
      throw new Error(response.error);
    }
  } catch (error) {
    responseError.textContent = `Error: ${error.message}`;
    responseError.classList.add('show');
  } finally {
    captureResponseBtn.disabled = false;
    captureResponseBtn.textContent = 'Capture Selected Text as Response';
  }
});

// Analyze hallucination
analyzeBtn.addEventListener('click', async () => {
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="loading"></span> Analyzing...';
  
  // Simulate analysis with random score
  await new Promise(resolve => setTimeout(resolve, 1500));
  
  const score = Math.floor(Math.random() * 10) + 1;
  const descriptions = [
    { max: 3, text: "Low hallucination risk. Response appears well-grounded.", color: "#10b981" },
    { max: 6, text: "Moderate hallucination risk. Some claims may need verification.", color: "#f59e0b" },
    { max: 10, text: "High hallucination risk. Significant discrepancies detected.", color: "#ef4444" }
  ];
  
  const desc = descriptions.find(d => score <= d.max);
  
  scoreValue.textContent = `${score}/10`;
  scoreValue.style.color = desc.color;
  scoreDescription.textContent = desc.text;
  resultCard.classList.add('show');
  
  analyzeBtn.disabled = false;
  analyzeBtn.textContent = 'Get Hallucination Level';
  
  // Scroll to result
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});

// Update prompt UI
function updatePromptUI(data) {
  if (data && data.text) {
    promptPreview.textContent = data.text;
    promptPreview.classList.remove('empty');
    promptStatus.textContent = 'Captured';
    promptStatus.classList.remove('status-empty');
    promptStatus.classList.add('status-captured');
    
    const date = new Date(data.timestamp);
    promptMeta.innerHTML = `
      <span>📏 ${data.length} characters</span>
      <span>🕒 ${date.toLocaleTimeString()}</span>
    `;
  }
}

// Update response UI
function updateResponseUI(data) {
  if (data && data.text) {
    responsePreview.textContent = data.text;
    responsePreview.classList.remove('empty');
    responseStatus.textContent = 'Captured';
    responseStatus.classList.remove('status-empty');
    responseStatus.classList.add('status-captured');
    
    const date = new Date(data.timestamp);
    responseMeta.innerHTML = `
      <span>📏 ${data.length} characters</span>
      <span>🕒 ${date.toLocaleTimeString()}</span>
    `;
  }
}

// Update analyze button state
function updateAnalyzeButton() {
  if (capturedPrompt && capturedResponse) {
    analyzeBtn.disabled = false;
  } else {
    analyzeBtn.disabled = true;
  }
}

// Initialize on load
initialize();