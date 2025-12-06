from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import openai
import os
from dotenv import load_dotenv
import re
import json

from rag_retriever import RAGRetriever, create_default_knowledge_base

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    """
    Extract math expression from prompt - only if it looks like a clear calculation
    More strict to avoid false positives
    """
    # Only look for math if prompt contains clear calculation keywords
    calc_keywords = ['calculate', 'what is', 'solve', 'compute', 'equals', 'equal to']
    if not any(keyword in prompt.lower() for keyword in calc_keywords):
        return None
    
    # More strict regex - require at least 2 numbers and an operator
    # Matches patterns like: 5 + 3, 10 * 2, 15.5 - 3.2
    m = re.search(r'(\d+(?:\.\d+)?\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?)*)', prompt)
    if m:
        return m.group(1).strip()
    return None


def safe_eval_expr(expr: str) -> Optional[float]:
    """Safely evaluate a mathematical expression"""
    try:
        # Remove whitespace and replace ^ with **
        cleaned = expr.replace(" ", "").replace("^", "**")
        
        # Only allow numbers and basic operators for safety
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
    use_rag: bool = False
    context: Optional[str] = None

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
    Uses OpenAI GPT to estimate hallucination score for a response.
    Returns a hallucination score from 0 to 10 and a confidence from 0 to 1.
    """
    prompt = f"""You are a hallucination detector. Your job is to identify if the Response contains false, unsupported, or made-up information compared to what was asked in the Prompt.

Given the following:

Prompt:
\"\"\"{premise}\"\"\"

Response:
\"\"\"{hypothesis}\"\"\"

Analyze if the Response contains hallucinations (false information, made-up facts, or claims not supported by the Prompt or general knowledge).

Return ONLY a valid JSON object with TWO keys (no markdown, no extra text):
- "hallucination_score": a number from 0 (no hallucination, completely accurate) to 10 (completely hallucinated, totally false)
- "confidence": a number from 0.0 to 1.0 representing your confidence in the score

Scoring guide:
- 0-2: Response is accurate and well-supported by facts
- 3-5: Some minor issues or unsupported claims
- 6-8: Significant false information or unsupported claims
- 9-10: Mostly or completely fabricated information (made-up names, places, facts)

Example: If asked "What is the third planet from the sun?" and the response invents fictional planets like "Cerulia" instead of saying "Earth", that would be a 10/10 hallucination.

Return format: {{"hallucination_score": X, "confidence": Y}}
"""

    try:
        print(f"\n{'='*60}")
        print(f"Calling OpenAI API...")
        print(f"Prompt preview: {premise[:100]}...")
        print(f"Response preview: {hypothesis[:100]}...")
        
        # Check if API key is set
        if not openai.api_key:
            raise ValueError("OpenAI API key not set. Check your .env file.")
        
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150  # Limit tokens since we only need JSON
        )

        text = completion.choices[0].message.content.strip()
        print(f"OpenAI raw output: {text}")

        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*|\s*```', '', text)
        
        # Extract JSON safely
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            json_str = match.group()
            print(f"Extracted JSON: {json_str}")
            result = json.loads(json_str)
            score = float(result.get("hallucination_score", 5.0))
            confidence = float(result.get("confidence", 0.5))
            
            # Validate ranges
            score = max(0.0, min(10.0, score))
            confidence = max(0.0, min(1.0, confidence))
            
            print(f"✓ Parsed successfully - Score: {score}/10, Confidence: {confidence}")
        else:
            print(f"✗ Warning: Could not extract JSON from response: {text}")
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
        print(f"✗ Unexpected error in compute_hhem_score: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = f"Unexpected error: {str(e)}"
        print(f"Raising HTTPException: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

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
        
        # ---------------- Quick arithmetic check ----------------
        expr = extract_math_from_prompt(request.prompt or "")
        print(f"Math expression check: {expr if expr else 'None detected'}")
        
        if expr:
            expected = safe_eval_expr(expr)
            print(f"Expected math result: {expected}")
            
            if expected is not None:
                # Look for answer after equals sign, or last number in response
                # Try patterns like "= 2", "is 2", "equals 2", or just the last number
                answer_patterns = [
                    r'=\s*(-?\d+(?:\.\d+)?)',  # After equals sign
                    r'is\s+(-?\d+(?:\.\d+)?)',  # After "is"
                    r'equals\s+(-?\d+(?:\.\d+)?)',  # After "equals"
                ]
                
                resp_val = None
                for pattern in answer_patterns:
                    match = re.search(pattern, request.response, re.IGNORECASE)
                    if match:
                        resp_val = float(match.group(1))
                        print(f"Found answer using pattern '{pattern}': {resp_val}")
                        break
                
                # If no pattern matched, get the last number in the response
                if resp_val is None:
                    all_numbers = re.findall(r'(-?\d+(?:\.\d+)?)', request.response)
                    if all_numbers:
                        resp_val = float(all_numbers[-1])
                        print(f"Using last number in response: {resp_val}")
                
                if resp_val is not None:
                    print(f"Expected: {expected}, Got: {resp_val}")
                    
                    if abs(resp_val - expected) < 1e-6:
                        print("✓ Math check PASSED - returning score 0.0")
                        return AnalysisResponse(
                            hallucination_score=0.0,
                            confidence=1.0,
                            raw_logits=[10.0, -10.0],
                            calibrated_score=0.0,
                            explanation=f"Simple arithmetic detected. Expression `{expr}` evaluates to {expected}, and response matches correctly.",
                            evidence=None,
                            rag_used=False
                        )
                    else:
                        print("✗ Math check FAILED - returning score 10.0")
                        return AnalysisResponse(
                            hallucination_score=10.0,
                            confidence=1.0,
                            raw_logits=[-10.0, 10.0],
                            calibrated_score=10.0,
                            explanation=f"Simple arithmetic detected. Expression `{expr}` evaluates to {expected}, but response contains {resp_val} (incorrect).",
                            evidence=None,
                            rag_used=False
                        )
        
        print("No simple math detected, proceeding to AI analysis...")

        # ---------------- Build enhanced premise ----------------
        evidence = None
        rag_used = False
        if request.use_rag and rag_retriever is not None:
            print("Retrieving evidence from RAG...")
            rag_used = True
            evidence = rag_retriever.retrieve(request.response, k=3)
            if evidence:
                evidence_text = " ".join([e.get('document', '') for e in evidence])
                enhanced_premise = f"{request.prompt}\n\nRelevant facts from knowledge base:\n{evidence_text}"
                print(f"✓ Retrieved {len(evidence)} documents")
            else:
                enhanced_premise = request.prompt
                print("No relevant documents found")
        else:
            enhanced_premise = request.prompt

        # ---------------- Compute hallucination ----------------
        try:
            result = compute_hhem_score(
                premise=enhanced_premise,
                hypothesis=request.response
            )
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            print(f"✗ Unexpected error in compute_hhem_score: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

        # Use GPT score directly as calibrated
        calibrated_score = result["hallucination_score"]

        # ---------------- Build explanation ----------------
        if calibrated_score <= 3.0:
            explanation = "✓ Low hallucination risk. The response appears well-grounded and accurate."
        elif calibrated_score <= 6.0:
            explanation = "⚠ Moderate hallucination risk. Some claims may need verification."
        else:
            explanation = "✗ High hallucination risk. Significant discrepancies or false information detected."
            
        if rag_used and evidence:
            explanation += f"\n\nRAG Analysis: Retrieved {len(evidence)} supporting documents from knowledge base."

        print(f"\n{'='*80}")
        print(f"FINAL RESULTS:")
        print(f"Score: {calibrated_score}/10")
        print(f"Confidence: {result['confidence']}")
        print(f"Explanation: {explanation}")
        print(f"{'='*80}\n")

        return AnalysisResponse(
            hallucination_score=result["hallucination_score"],
            confidence=result["confidence"],
            raw_logits=result["raw_logits"],
            calibrated_score=calibrated_score,
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
    api_key_set = bool(openai.api_key and openai.api_key != "")
    return {
        "status": "healthy",
        "rag_initialized": rag_retriever is not None,
        "openai_api_key_set": api_key_set,
        "openai_api_key_preview": f"{openai.api_key[:10]}..." if api_key_set else "Not set"
    }

# Test OpenAI endpoint
@app.get("/test-openai")
async def test_openai():
    """Test if OpenAI API is working"""
    try:
        if not openai.api_key:
            return {"status": "error", "message": "OpenAI API key not set"}
        
        print("Testing OpenAI API connection...")
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test successful' in JSON format with a key 'message'"}],
            temperature=0,
            max_tokens=50
        )
        
        response_text = completion.choices[0].message.content
        print(f"OpenAI test response: {response_text}")
        
        return {
            "status": "success",
            "message": "OpenAI API is working",
            "response": response_text
        }
    except Exception as e:
        print(f"OpenAI test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }


# Run with: uvicorn backend:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
