from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import openai
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import json

from rag_retriever import RAGRetriever, create_default_knowledge_base

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Hallucination Detector API")

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


def safe_eval_expr(expr: str) -> Optional[float]:
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

# ---------------- RAG retriever ----------------
rag_retriever = None

@app.on_event("startup")
async def startup_event():
    global rag_retriever
    try:
        rag_retriever = create_default_knowledge_base()
        print("✓ RAG retriever initialized")
    except Exception as e:
        print(f"✗ Failed to initialize RAG: {e}")
        rag_retriever = None

# ---------------- Pydantic Models ----------------
class AnalysisRequest(BaseModel):
    prompt: str
    response: str
    use_rag: bool = True  # Default to True since we need RAG
    context: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

class AnalysisResponse(BaseModel):
    hallucination_score: float
    confidence: float
    raw_logits: List[float]
    calibrated_score: Optional[float] = None
    evidence: Optional[List[Dict]] = None
    explanation: str
    rag_used: bool = False

# ---------------- OpenAI-based hallucination score ----------------
def compute_hhem_score(premise: str, hypothesis: str):
    """
    Uses OpenAI GPT to estimate hallucination score with RAG evidence.
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
        print(f"Calling OpenAI API...")
        print(f"Prompt preview: {premise[:100]}...")
        print(f"Response preview: {hypothesis[:100]}...")
        
        if not openai.api_key:
            raise ValueError("OpenAI API key not set")
        
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150  # Limit tokens since we only need JSON
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

        # Construct raw_logits for reference
        raw_logits = [score, 10 - score]

        return {
            "hallucination_score": score,
            "confidence": confidence,
            "raw_logits": raw_logits
        }

    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error: {e}")
        print(f"Attempted to parse: {text if 'text' in locals() else 'N/A'}")
        error_msg = f"Failed to parse OpenAI response as JSON: {str(e)}"
        print(f"Raising HTTPException: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    except openai.OpenAIError as e:
        print(f"✗ OpenAI API Error: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = f"OpenAI API error: {str(e)}"
        print(f"Raising HTTPException: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    except Exception as e:
        print(f"✗ Gemini Judge Error: {str(e)}")
        # Fallback to OpenAI score if Gemini fails
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
        print(f"Use RAG: {request.use_rag}")
        print(f"{'='*80}")

        # Validate temperature
        if not (0.0 <= request.temperature <= 2.0):
            raise HTTPException(
                status_code=400,
                detail="Temperature must be between 0.0 and 2.0"
            )
        
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
                            rag_used=False
                        )
                    else:
                        return AnalysisResponse(
                            hallucination_score=10.0,
                            confidence=1.0,
                            raw_logits=[-10.0, 10.0],
                            calibrated_score=10.0,
                            explanation=f"Arithmetic error. Expected {expected}, got {resp_val} ✗",
                            evidence=None,
                            rag_used=False
                        )

        # ---------------- RAG Evidence Retrieval ----------------
        evidence = None
        evidence_text = ""
        rag_used = False
        
        if rag_retriever is not None:
            print("Retrieving evidence from RAG...")
            rag_used = True
            evidence = rag_retriever.retrieve(request.response, k=3)
            
            if evidence:
                evidence_text = "\n".join([
                    f"{i+1}. {e.get('document', '')} (Score: {e.get('score', 0):.2f})"
                    for i, e in enumerate(evidence)
                ])
                print(f"✓ Retrieved {len(evidence)} documents")
            else:
                print("No relevant documents found")

        # ---------------- OpenAI Analysis ----------------
        try:
            result = compute_hhem_score(
                premise=enhanced_premise,
                hypothesis=request.response
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
        
        if final_score <= 3.0:
            base_explanation = "✓ Low hallucination risk. Response appears accurate."
        elif final_score <= 6.0:
            base_explanation = "⚠ Moderate hallucination risk. Some claims need verification."
        else:
            base_explanation = "✗ High hallucination risk. Significant issues detected."
        
        # Add judge insight
        if judge_result['agreement'] == 'adjusted':
            judge_note = f"\n\n🔍 LLM Judge: Adjusted from OpenAI's {openai_result['hallucination_score']}/10 to {final_score}/10"
        else:
            judge_note = f"\n\n🔍 LLM Judge: Confirmed OpenAI's assessment"
        
        explanation = base_explanation + judge_note

        if rag_used and evidence:
            explanation += f"\n\n📚 Evidence: {len(evidence)} supporting documents analyzed"

        print(f"\n{'='*80}")
        print(f"FINAL RESULTS:")
        print(f"Score: {calibrated_score}/10")
        print(f"Confidence: {result['confidence']}")
        print(f"Explanation: {explanation}")
        print(f"{'='*80}\n")

        return AnalysisResponse(
            hallucination_score=final_score,
            confidence=judge_result['confidence'],
            raw_logits=openai_result['raw_logits'],
            calibrated_score=final_score,
            explanation=explanation,
            evidence=evidence if rag_used else None,
            rag_used=rag_used
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
        "openai_api_key_set": openai_key_set,
        "gemini_api_key_set": gemini_key_set
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)