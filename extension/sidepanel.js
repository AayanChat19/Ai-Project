// Side panel JavaScript for Hallucination Detector with Evidence Display

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
const evidenceContainer = document.getElementById('evidenceContainer');

// State
let capturedPrompt = null;
let capturedResponse = null;

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
  checkAPIHealth();
}

async function checkAPIHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();
    console.log('API Health:', data);
    
    if (!data.openai_api_key_set) {
      console.warn('OpenAI API key not set');
    }
    if (!data.gemini_api_key_set) {
      console.warn('Gemini API key not set');
    }
  } catch (error) {
    console.warn('Backend API not available:', error.message);
  }
}

// Capture prompt
capturePromptBtn.addEventListener('click', async () => {
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

// Analyze hallucination
analyzeBtn.addEventListener('click', async () => {
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="loading"></span> Analyzing...';
  resultCard.classList.remove('show');
  
  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: capturedPrompt.text,
        response: capturedResponse.text,
        use_rag: true
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `API error: ${response.status}`);
    }
    
    const result = await response.json();
    displayResults(result);
    await saveAnalysisToStorage(result);
    
  } catch (error) {
    console.error('Analysis error:', error);
    alert(`Analysis failed: ${error.message}\n\nMake sure the backend is running:\npython backend.py`);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Get Hallucination Level';
  }
});

function displayResults(result) {
  const score = result.calibrated_score ?? result.hallucination_score;

  // Determine color
  let color;
  if (score <= 3.0) {
    color = "#10b981"; // Green
  } else if (score <= 6.0) {
    color = "#f59e0b"; // Orange
  } else {
    color = "#ef4444"; // Red
  }

  // Update score
  scoreValue.textContent = `${score.toFixed(1)}/10`;
  scoreValue.style.color = color;

  // Build description
  let description = result.explanation || "No explanation available.";

  // Add judge information
  if (result.judge_explanation) {
    description += `\n\n🔍 Judge Reasoning:\n${result.judge_explanation}`;
  }

  // Add scores breakdown
  if (result.openai_score !== undefined && result.gemini_score !== undefined) {
    description += `\n\n📊 Scores:\nOpenAI: ${result.openai_score.toFixed(1)}/10\nGemini: ${result.gemini_score.toFixed(1)}/10`;
  }

  if (typeof result.confidence === "number") {
    description += `\n\nConfidence: ${(result.confidence * 100).toFixed(1)}%`;
  }

  scoreDescription.textContent = description;
  scoreDescription.style.whiteSpace = 'pre-line';

  // Display evidence
  displayEvidence(result.evidence);

  // Show result card
  resultCard.classList.add('show');

  setTimeout(() => {
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 100);

  console.log('Analysis Results:', result);
}

function displayEvidence(evidence) {
  evidenceContainer.innerHTML = '';
  
  if (!evidence || evidence.length === 0) {
    evidenceContainer.innerHTML = '<div class="evidence-empty">No evidence retrieved</div>';
    return;
  }

  const evidenceTitle = document.createElement('div');
  evidenceTitle.className = 'evidence-title';
  evidenceTitle.textContent = '📚 Supporting Evidence';
  evidenceContainer.appendChild(evidenceTitle);

  evidence.forEach((item, index) => {
    const evidenceItem = document.createElement('div');
    evidenceItem.className = 'evidence-item';
    
    const evidenceRank = document.createElement('div');
    evidenceRank.className = 'evidence-rank';
    evidenceRank.textContent = `Evidence #${index + 1}`;
    
    const evidenceText = document.createElement('div');
    evidenceText.className = 'evidence-text';
    evidenceText.textContent = item.document;
    
    const evidenceMeta = document.createElement('div');
    evidenceMeta.className = 'evidence-meta';
    
    const source = item.metadata?.source || 'Unknown Source';
    const topic = item.metadata?.topic || 'General';
    const url = item.metadata?.url || null;
    const relevance = item.score ? `Relevance: ${(1 / (1 + item.score)).toFixed(3)}` : '';
    
    // Create clickable source link if URL exists
    let sourceHTML = `<span class="evidence-source">📖 ${source}</span>`;
    if (url) {
      sourceHTML = `<a href="${url}" target="_blank" class="evidence-source evidence-link" title="Open source">📖 ${source} 🔗</a>`;
    }
    
    evidenceMeta.innerHTML = `
      ${sourceHTML}
      <span class="evidence-topic">🏷️ ${topic}</span>
      ${relevance ? `<span class="evidence-relevance">${relevance}</span>` : ''}
    `;
    
    evidenceItem.appendChild(evidenceRank);
    evidenceItem.appendChild(evidenceText);
    evidenceItem.appendChild(evidenceMeta);
    evidenceContainer.appendChild(evidenceItem);
  });
}

async function saveAnalysisToStorage(result) {
  const timestamp = Date.now();
  const scoreToStore = result.calibrated_score ?? result.hallucination_score;

  const analysisRecord = {
    timestamp,
    prompt: capturedPrompt.text.substring(0, 500),
    response: capturedResponse.text.substring(0, 500),
    score: scoreToStore,
    confidence: result.confidence,
    openai_score: result.openai_score,
    gemini_score: result.gemini_score,
    evidence_count: result.evidence?.length || 0,
    explanation: result.explanation
  };
  
  const data = await chrome.storage.local.get(['analysisHistory']);
  const history = data.analysisHistory || [];
  history.push(analysisRecord);
  
  if (history.length > 100) {
    history.shift();
  }
  
  await chrome.storage.local.set({ analysisHistory: history });
  console.log('Analysis saved to history');
}

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

function updateAnalyzeButton() {
  analyzeBtn.disabled = !(capturedPrompt && capturedResponse);
}

initialize();