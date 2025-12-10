from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import openai
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import json

# Import the enhanced RAG retriever
from rag_retriever import EnhancedRAGRetriever, create_enhanced_knowledge_base

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Global RAG retriever
rag_retriever = None

# Modern lifespan event handler (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize RAG retriever
    global rag_retriever
    try:
        print("\n" + "="*60)
        print("🚀 Starting up Hallucination Detector API...")
        print("="*60)
        
        rag_retriever = create_enhanced_knowledge_base(
            enable_web_search=True,
            relevance_threshold=0.65
        )
        print("✓ Enhanced RAG retriever with web search initialized")
        print("="*60 + "\n")
    except Exception as e:
        print(f"✗ Failed to initialize RAG: {e}")
        rag_retriever = None
    
    yield  # Application runs here
    
    # Shutdown: Cleanup (if needed)
    print("\n" + "="*60)
    print("🛑 Shutting down Hallucination Detector API...")
    print("="*60 + "\n")

# Create FastAPI app with lifespan handler
app = FastAPI(
    title="Hallucination Detector API with Web Search",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Safe math evaluator ----------------
def extract_math_from_prompt(prompt: str) -> Optional[str]:
    """Extract math expression from prompt"""
    calc_keywords = ['calculate', 'what is', 'solve', 'compute', 'equals', 'equal to']
    if not any(keyword in prompt.lower() for keyword in calc_keywords):
        return None
    
    m = re.search(r'(\d+(?:\.\d+)?\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?)*)', prompt)
    if m:
        return m.group(1).strip()
    return None


# ---------------- Safe math evaluator ----------------
def extract_math_from_prompt(prompt: str) -> Optional[str]:
    """Safely evaluate a mathematical expression"""
    try:
        cleaned = expr.replace(" ", "").replace("^", "**")
        if not re.match(r'^[\d\.\+\-\*\/\(\)]+$', cleaned):
            return None
        result = eval(cleaned)
        return float(result)
    except Exception as e:
        print(f"Math eval error: {e}")
        return None

# ---------------- Pydantic Models ----------------
class AnalysisRequest(BaseModel):
    prompt: str
    response: str
    use_rag: bool = True
    use_web_search: bool = True  # New: control web search
    context: Optional[str] = None

class AnalysisResponse(BaseModel):
    hallucination_score: float
    confidence: float
    raw_logits: List[float]
    calibrated_score: Optional[float] = None
    evidence: Optional[List[Dict]] = None
    explanation: str
    rag_used: bool = False
    web_search_used: bool = False  # New: indicates if web search was used
    judge_explanation: Optional[str] = None
    openai_score: Optional[float] = None
    gemini_score: Optional[float] = None
    analysis_style: Optional[str] = None

# ---------------- OpenAI-based hallucination score ----------------
def compute_openai_score(premise: str, hypothesis: str, evidence_text: str = ""):
    """
    Uses OpenAI GPT to estimate hallucination score with RAG evidence.
    Always uses temperature=0 for consistency.
    """
    evidence_section = ""
    if evidence_text:
        evidence_section = f"\n\nRelevant Evidence from Knowledge Base:\n{evidence_text}\n"
    
    prompt = f"""You are a hallucination detector. Analyze if the Response contains false, unsupported, or made-up information.

Prompt:
\"\"\"{premise}\"\"\"

Response:
\"\"\"{hypothesis}\"\"\"{evidence_section}

Analyze if the Response contains hallucinations (false information, made-up facts, or unsupported claims).
Use the evidence provided to verify claims.

Return ONLY a valid JSON object with THREE keys:
- "hallucination_score": number from 0 (accurate) to 10 (completely false)
- "confidence": number from 0.0 to 1.0
- "reasoning": brief explanation of the score

Scoring guide:
- 0-2: Response is accurate and well-supported
- 3-5: Some minor issues or unsupported claims
- 6-8: Significant false information
- 9-10: Mostly fabricated information

Format: {{"hallucination_score": X, "confidence": Y, "reasoning": "explanation"}}
"""

    try:
        print(f"\n{'='*60}")
        print(f"Calling OpenAI API (temperature: 0.0 for consistency)...")
        
        if not openai.api_key:
            raise ValueError("OpenAI API key not set")
        
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )

        text = completion.choices[0].message.content.strip()
        print(f"OpenAI raw output: {text}")

        text = re.sub(r'```json\s*|\s*```', '', text)
        match = re.search(r'\{[^{}]*\}', text)
        
        if match:
            json_str = match.group()
            result = json.loads(json_str)
            score = float(result.get("hallucination_score", 5.0))
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "No reasoning provided")
            
            score = max(0.0, min(10.0, score))
            confidence = max(0.0, min(1.0, confidence))
            
            print(f"✓ OpenAI Score: {score}/10, Confidence: {confidence}")
            return {
                "hallucination_score": score,
                "confidence": confidence,
                "reasoning": reasoning,
                "raw_logits": [score, 10 - score]
            }
        else:
            raise ValueError("No valid JSON found in API response")

    except Exception as e:
        print(f"✗ OpenAI API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# ---------------- Calculate Analysis Style ----------------
def calculate_analysis_style(confidence: float, score_difference: float) -> str:
    """Determine if the analysis was Deterministic, Balanced, or Creative"""
    if confidence >= 0.8 and score_difference <= 1.0:
        return "Deterministic"
    elif confidence < 0.6 or score_difference > 3.0:
        return "Creative"
    else:
        return "Balanced"

def gemini_judge(premise: str, hypothesis: str, openai_result: Dict, evidence: List[Dict]) -> Dict:
    """Uses Gemini as LLM Judge to validate OpenAI's score."""
    evidence_text = "\n".join([
        f"- [{e.get('source_type', 'unknown').upper()}] {e.get('document', '')}" 
        for e in evidence
    ]) if evidence else "No evidence available"
    
    prompt = f"""You are an LLM Judge evaluating a hallucination detection result.

IMPORTANT: Use the SAME scale as the OpenAI detector: 0 means the Response is fully ACCURATE and 10 means the Response is COMPLETELY FABRICATED/UNSUPPORTED.

Return EXACTLY ONE JSON OBJECT (no surrounding text, no code fences) with these fields:
{{
  "final_score": <number 0-10>,
  "agreement": <"agree" or "adjusted">,
  "judge_reasoning": "<short explanation (1-3 sentences)>",
  "confidence": <number 0.0-1.0>
}}

Original Prompt:
\"\"\"{premise}\"\"\"

Response Being Evaluated:
\"\"\"{hypothesis}\"\"\"

Evidence (includes both local knowledge base and web sources):
{evidence_text}

OpenAI's Analysis:
- Score: {openai_result['hallucination_score']}/10
- Confidence: {openai_result['confidence']}
- Reasoning: {openai_result.get('reasoning', 'N/A')}

Your task: Review this analysis and either AGREE or ADJUST the score based on:
1. Does the evidence support or contradict the response?
2. Is OpenAI's score reasonable?
3. Are there any missed hallucinations or false positives?
"""

    try:
        print(f"\n{'='*60}")
        print(f"Calling Gemini Judge...")
        
        try:
            model = genai.GenerativeModel('models/gemini-2.5-flash')
        except:
            try:
                model = genai.GenerativeModel('gemini-pro')
            except:
                model = genai.GenerativeModel('gemini-1.0-pro')
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        print(f"Gemini raw output: {text}")
        
        text = re.sub(r'```json\s*|\s*```', '', text)
        match = re.search(r'\{[^{}]*\}', text)
        
        if match:
            json_str = match.group()
            result = json.loads(json_str)
            
            final_score = float(result.get("final_score", openai_result['hallucination_score']))
            agreement = result.get("agreement", "agree")
            judge_reasoning = result.get("judge_reasoning", "No reasoning provided")
            confidence = float(result.get("confidence", openai_result['confidence']))
            
            final_score = max(0.0, min(10.0, final_score))
            confidence = max(0.0, min(1.0, confidence))
            
            print(f"✓ Gemini Judge: {final_score}/10 ({agreement})")
            
            return {
                "final_score": final_score,
                "agreement": agreement,
                "judge_reasoning": judge_reasoning,
                "confidence": confidence
            }
        else:
            raise ValueError("No valid JSON from Gemini")
            
    except Exception as e:
        print(f"✗ Gemini Judge Error: {str(e)}")
        return {
            "final_score": openai_result['hallucination_score'],
            "agreement": "fallback",
            "judge_reasoning": f"Gemini judge unavailable: {str(e)}",
            "confidence": openai_result['confidence']
        }

# ---------------- API Endpoint ----------------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_hallucination(request: AnalysisRequest):
    try:
        print(f"\n{'='*80}")
        print(f"NEW ANALYSIS REQUEST")
        print(f"{'='*80}")
        print(f"Prompt: {request.prompt[:150]}...")
        print(f"Response: {request.response[:150]}...")
        print(f"Web Search Enabled: {request.use_web_search}")
        print(f"{'='*80}")

        # ---------------- Quick arithmetic check ----------------
        expr = extract_math_from_prompt(request.prompt or "")
        
        if expr:
            expected = safe_eval_expr(expr)
            if expected is not None:
                answer_patterns = [
                    r'=\s*(-?\d+(?:\.\d+)?)',
                    r'is\s+(-?\d+(?:\.\d+)?)',
                    r'equals\s+(-?\d+(?:\.\d+)?)',
                ]
                
                resp_val = None
                for pattern in answer_patterns:
                    match = re.search(pattern, request.response, re.IGNORECASE)
                    if match:
                        resp_val = float(match.group(1))
                        break
                
                if resp_val is None:
                    all_numbers = re.findall(r'(-?\d+(?:\.\d+)?)', request.response)
                    if all_numbers:
                        resp_val = float(all_numbers[-1])
                
                if resp_val is not None:
                    if abs(resp_val - expected) < 1e-6:
                        return AnalysisResponse(
                            hallucination_score=0.0,
                            confidence=1.0,
                            raw_logits=[10.0, -10.0],
                            calibrated_score=0.0,
                            explanation=f"Simple arithmetic verified. Expression `{expr}` = {expected} ✓",
                            evidence=None,
                            rag_used=False,
                            web_search_used=False,
                            openai_score=0.0,
                            gemini_score=0.0,
                            analysis_style="Deterministic"
                        )
                    else:
                        return AnalysisResponse(
                            hallucination_score=10.0,
                            confidence=1.0,
                            raw_logits=[-10.0, 10.0],
                            calibrated_score=10.0,
                            explanation=f"Arithmetic error. Expected {expected}, got {resp_val} ✗",
                            evidence=None,
                            rag_used=False,
                            web_search_used=False,
                            openai_score=10.0,
                            gemini_score=10.0,
                            analysis_style="Deterministic"
                        )

        # ---------------- Enhanced RAG Evidence Retrieval with Web Search ----------------
        evidence = None
        evidence_text = ""
        rag_used = False
        web_search_used = False
        
        if rag_retriever is not None and request.use_rag:
            print("Retrieving evidence from Enhanced RAG (with web search fallback)...")
            rag_used = True
            
            # Temporarily disable web search if requested
            original_web_setting = rag_retriever.enable_web_search
            if not request.use_web_search:
                rag_retriever.enable_web_search = False
            
            try:
                # IMPORTANT: Search using the PROMPT, not the response
                # This finds evidence relevant to what was asked, not what was claimed
                search_query = request.prompt
                
                # For specific factual queries, extract the key question
                # e.g., "what is the aqi index of delhi" -> good search query
                evidence = rag_retriever.retrieve(search_query, k=5)
                
                # Check if web search was used
                if evidence:
                    web_search_used = any(e.get('source_type') == 'web' for e in evidence)
                    
                    evidence_text = "\n".join([
                        f"{i+1}. [{e.get('source_type', 'unknown').upper()}] {e.get('document', '')} (Relevance: {e.get('relevance', 0):.2f})"
                        for i, e in enumerate(evidence)
                    ])
                    
                    local_count = sum(1 for e in evidence if e.get('source_type') == 'local')
                    web_count = sum(1 for e in evidence if e.get('source_type') == 'web')
                    
                    print(f"✓ Retrieved {len(evidence)} documents (Local: {local_count}, Web: {web_count})")
                else:
                    print("No relevant documents found")
            finally:
                # Restore original web search setting
                rag_retriever.enable_web_search = original_web_setting

        # ---------------- OpenAI Analysis ----------------
        try:
            openai_result = compute_openai_score(
                premise=request.prompt,
                hypothesis=request.response,
                evidence_text=evidence_text
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI analysis failed: {str(e)}")

        # ---------------- Gemini LLM Judge ----------------
        judge_result = gemini_judge(
            premise=request.prompt,
            hypothesis=request.response,
            openai_result=openai_result,
            evidence=evidence or []
        )

        # ---------------- Build explanation ----------------
        final_score = judge_result['final_score']
        
        score_difference = abs(openai_result['hallucination_score'] - final_score)
        analysis_style = calculate_analysis_style(
            confidence=judge_result['confidence'],
            score_difference=score_difference
        )
        
        if final_score <= 3.0:
            base_explanation = "✓ Low hallucination risk. Response appears accurate."
        elif final_score <= 6.0:
            base_explanation = "⚠ Moderate hallucination risk. Some claims need verification."
        else:
            base_explanation = "✗ High hallucination risk. Significant issues detected."
        
        # Add judge insight
        if judge_result['agreement'] == 'adjusted':
            judge_note = f"\n\n🔍 LLM Judge: Adjusted from OpenAI's {openai_result['hallucination_score']}/10 to {final_score}/10"
        elif judge_result['agreement'] == 'fallback':
            judge_note = f"\n\n⚠️ LLM Judge: Unavailable, using OpenAI score only"
        else:
            judge_note = f"\n\n🔍 LLM Judge: Confirmed OpenAI's assessment"
        
        explanation = base_explanation + judge_note + "\n\n" + openai_result.get('reasoning', '')

        if rag_used and evidence:
            local_count = sum(1 for e in evidence if e.get('source_type') == 'local')
            web_count = sum(1 for e in evidence if e.get('source_type') == 'web')
            
            if web_search_used:
                explanation += f"\n\n📚 Evidence: {local_count} local + 🌐 {web_count} web sources analyzed"
            else:
                explanation += f"\n\n📚 Evidence: {len(evidence)} local documents analyzed"

        print(f"\n{'='*80}")
        print(f"FINAL RESULTS:")
        print(f"OpenAI Score: {openai_result['hallucination_score']}/10")
        print(f"Gemini Score: {final_score}/10")
        print(f"Agreement: {judge_result['agreement']}")
        print(f"Analysis Style: {analysis_style}")
        print(f"Web Search Used: {web_search_used}")
        print(f"{'='*80}\n")

        return AnalysisResponse(
            hallucination_score=final_score,
            confidence=judge_result['confidence'],
            raw_logits=openai_result['raw_logits'],
            calibrated_score=final_score,
            explanation=explanation,
            evidence=evidence if rag_used else None,
            rag_used=rag_used,
            web_search_used=web_search_used,
            judge_explanation=judge_result['judge_reasoning'],
            openai_score=openai_result['hallucination_score'],
            gemini_score=final_score,
            analysis_style=analysis_style
        )

    except Exception as e:
        print(f"✗ Unexpected error in /analyze: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    openai_key_set = bool(openai.api_key and openai.api_key != "")
    gemini_key_set = bool(os.getenv("GEMINI_API_KEY"))
    
    return {
        "status": "healthy",
        "rag_initialized": rag_retriever is not None,
        "web_search_enabled": rag_retriever.enable_web_search if rag_retriever else False,
        "openai_api_key_set": openai_key_set,
        "gemini_api_key_set": gemini_key_set
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)