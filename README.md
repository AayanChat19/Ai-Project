# Enhanced Hallucination Detector

A comprehensive AI hallucination detection system with RAG-enhanced verification, claim-level analysis, and multi-source evidence retrieval. Built with FastAPI backend and Chrome extension frontend.

## Features

- **Multi-Model Analysis**: Combines OpenAI GPT-4 and Google Gemini for robust hallucination detection
- **RAG-Enhanced Verification**: Local knowledge base + web search for evidence-based validation
- **Claim-Level Analysis**: Extracts and verifies individual factual claims
- **Smart Evidence Retrieval**: Multi-query retrieval with reliability scoring
- **Web Search Integration**: Automatic fallback to Wikipedia, Google, and academic sources
- **Source Reliability Ranking**: Prioritizes evidence from high-quality sources
- **Chrome Extension**: Easy-to-use browser interface for real-time verification

## Project Structure

```
hallucination-detector/
├── backend/
│   ├── main.py                 # FastAPI server with detection logic
│   ├── rag_retriever.py        # Enhanced RAG with web search
│   ├── requirements.txt        # Python dependencies
|   ├── create_10.py            # creates test cases json
|   ├── eval (1-5)              # baseline vs model evaluation
│   └── .env                    # API keys (not in repo)
├── extension/
│   ├── manifest.json           # Chrome extension config
│   ├── popup.html             # Extension UI
│   ├── popup.css              # Styling
│   └── popup.js               # Frontend logic

```

##  Quick Start

### Prerequisites

- Python 3.8+
- Node.js (for any frontend tooling)
- API Keys:
  - OpenAI API key
  - Google Gemini API key
  - Serper API key (optional, for Google search)

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd hallucination-detector/backend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the `backend/` directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
SERPER_API_KEY=your_serper_api_key_here  # Optional
```

5. **Run the server**
```bash
python backend.py
```

The API will be available at `http://localhost:8000`

### Chrome Extension Setup

1. **Open Chrome Extensions**
   - Navigate to `chrome://extensions/`
   - Enable "Developer mode" (top right)

2. **Load Extension**
   - Click "Load unpacked"
   - Select the `extension/` folder

3. **Use Extension**
   - Click the extension icon in your browser
   - Paste prompt and response to analyze
   - View hallucination score and evidence

## Architecture

### Detection Pipeline

1. **Evidence Retrieval**
   - Multi-query search (prompt + response)
   - Local knowledge base retrieval
   - Web search fallback with reliability ranking

2. **Claim Extraction**
   - GPT-4 extracts individual factual claims
   - Filters for verifiable statements

3. **Claim Verification**
   - Semantic similarity matching with evidence
   - Deep search for unsupported claims

4. **Scoring**
   - OpenAI GPT-4 provides initial assessment
   - Gemini 2.5 acts as LLM judge
   - Combined scoring with confidence metrics

### RAG System

- **Embedding Model**: `all-MiniLM-L6-v2`
- **Vector Store**: FAISS with L2 distance
- **Knowledge Base**: 100+ curated facts across multiple domains
- **Web Sources**: Wikipedia (0.95), Google (0.90), Semantic Scholar (0.85)

### Tunable Parameters

In `rag_retriever.py`:
- `relevance_threshold`: Minimum similarity for local results (default: 0.7)
- `enable_web_search`: Toggle web search fallback (default: True)

In `main.py`:
- `k`: Number of evidence documents to retrieve (default: 5)
- `max_results`: Web search results limit (default: 3-5)
