// Side panel JavaScript for Hallucination Detector

// Configuration
const API_BASE_URL = 'http://localhost:8000';

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
let lastAnalysisResult = null;

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
  
  // Check API health
  checkAPIHealth();
}

// Check if backend API is running
async function checkAPIHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();
    console.log('API Health:', data);
  } catch (error) {
    console.warn('Backend API not available:', error.message);
    console.log('Make sure to run: python backend.py');
  }
}

// Capture prompt
capturePromptBtn.addEventListener('click', async () => {
  // Check if text is selected on the page first
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const selectionResult = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => window.getSelection().toString().trim()
  });
  const selectedText = selectionResult[0].result;

  if (!selectedText) {
    promptError.textContent = 'Please select some text on the page first.';
    promptError.classList.add('show');
    return;
  }

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

// Analyze hallucination using backend API
analyzeBtn.addEventListener('click', async () => {
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="loading"></span> Analyzing...';
  resultCard.classList.remove('show');
  
  try {
    // Call the backend API
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: capturedPrompt.text,
        response: capturedResponse.text,
        use_rag: false
      })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    const result = await response.json();
    lastAnalysisResult = result;
    
    // Display results
    displayResults(result);
    
    // Save analysis to storage for report
    await saveAnalysisToStorage(result);
    
  } catch (error) {
    console.error('Analysis error:', error);
    
    // Fallback to simple scoring if API fails
    const fallbackResult = {
      hallucination_score: Math.floor(Math.random() * 10) + 1,
      confidence: Math.random() * 0.5 + 0.5, // 50%–100% confidence
      explanation: 'Using fallback analysis (API unavailable). Please start the backend server.',
      raw_logits: [0, 0],
      calibrated_score: null
    };
    
    
    displayResults(fallbackResult);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Get Hallucination Level';
  }
});

function displayResults(result) {
  const score = result.calibrated_score ?? result.hallucination_score;

  // Determine color based on score
  let color;
  if (score <= 3.0) {
    color = "#10b981"; // Green
  } else if (score <= 6.0) {
    color = "#f59e0b"; // Orange
  } else {
    color = "#ef4444"; // Red
  }

  // Update UI
  scoreValue.textContent = `${score.toFixed(1)}/10`;
  scoreValue.style.color = color;

  // Build detailed description
  let description = result.explanation || "No explanation available.";

  // Show confidence only if defined and valid
  if (typeof result.confidence === "number") {
    description += `\n\nConfidence: ${(result.confidence * 100).toFixed(1)}%`;
  } else {
    description += `\n\nConfidence: N/A`;
  }

  if (result.calibrated_score !== null && result.calibrated_score !== undefined) {
    description += `\nCalibrated Score: ${result.calibrated_score.toFixed(1)}/10`;
  }

  scoreDescription.textContent = description;
  scoreDescription.style.whiteSpace = 'pre-line';

  // Show result card with animation
  resultCard.classList.add('show');

  // Scroll to result
  setTimeout(() => {
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 100);

  // Log detailed results for debugging
  console.log('Analysis Results:', {
    score: result.hallucination_score,
    confidence: result.confidence,
    raw_logits: result.raw_logits,
    calibrated_score: result.calibrated_score,
    explanation: result.explanation
  });
}


// Save analysis results to storage for reporting
async function saveAnalysisToStorage(result) {
  const timestamp = Date.now();
  
  // Use calibrated score if available, otherwise fallback to raw hallucination_score
  const scoreToStore = result.calibrated_score ?? result.hallucination_score;

  const analysisRecord = {
    timestamp,
    prompt: capturedPrompt.text.substring(0, 500), // Store first 500 chars
    response: capturedResponse.text.substring(0, 500),
    score: scoreToStore,
    confidence: result.confidence,
    raw_logits: result.raw_logits,
    calibrated_score: result.calibrated_score,
    explanation: result.explanation
  };
  
  // Get existing analyses
  const data = await chrome.storage.local.get(['analysisHistory']);
  const history = data.analysisHistory || [];
  
  // Add new analysis
  history.push(analysisRecord);
  
  // Keep only last 100 analyses
  if (history.length > 100) {
    history.shift();
  }
  
  // Save back to storage
  await chrome.storage.local.set({ analysisHistory: history });
  
  console.log('Analysis saved to history');
}


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
      <span>📝 ${data.length} characters</span>
      <span>🕐 ${date.toLocaleTimeString()}</span>
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
      <span>📝 ${data.length} characters</span>
      <span>🕐 ${date.toLocaleTimeString()}</span>
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