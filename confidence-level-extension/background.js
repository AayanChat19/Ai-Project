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
    m = re.search(r'([0-9\.\s\+\-\*\/\^\(\)]+)', prompt)
    if m and re.search(r'[0-9]', m.group(1)):
        return m.group(1).strip()
    return None

def safe_eval_expr(expr: str) -> Optional[float]:
    try:
        cleaned = expr.replace(",", "").strip().replace("^", "**")
        return float(eval(cleaned))
    except Exception:
        return None

# ---------------- RAG retriever ----------------
rag_retriever = None

@app.on_event("startup")
async def startup_event():
    global rag_retriever
    try:
        rag_retriever = create_default_knowledge_base()
        print("RAG retriever initialized")
    except Exception as e:
        print(f"Failed to initialize RAG: {e}")
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
    prompt = f"""
You are a hallucination detector.

Given the following:

Prompt:
\"\"\"{premise}\"\"\"

Response:
\"\"\"{hypothesis}\"\"\"

Return ONLY a JSON object with TWO keys:
- "hallucination_score": a number from 0 (no hallucination) to 10 (fully hallucinated)
- "confidence": a number from 0.0 to 1.0 representing your confidence in the score

Do NOT include any extra text or explanation. Do NOT break JSON formatting.
"""

    try:
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-4 / gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = completion.choices[0].message.content.strip()
        print(f"OpenAI raw output: {text}")

        # Extract JSON safely using regex to allow minor formatting issues
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            score = float(result.get("hallucination_score", 0))
            confidence = float(result.get("confidence", 0.5))
        else:
            print(f"Warning: No JSON found in model response: {text}")
            score = 0.0
            confidence = 0.5

        # Construct raw_logits for reference, not used for calibration anymore
        raw_logits = [score, 10 - score]

        return {
            "hallucination_score": score,
            "confidence": confidence,
            "raw_logits": raw_logits
        }

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {
            "hallucination_score": 0.0,
            "confidence": 0.5,
            "raw_logits": [0.0, 10.0]
        }

# ---------------- API Endpoint ----------------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_hallucination(request: AnalysisRequest):
    try:
        # ---------------- Quick arithmetic check ----------------
        expr = extract_math_from_prompt(request.prompt or "")
        if expr:
            expected = safe_eval_expr(expr)
            if expected is not None:
                num_match = re.search(r'(-?\d+(\.\d+)?)', request.response)
                if num_match:
                    resp_val = float(num_match.group(1))
                    if abs(resp_val - expected) < 1e-6:
                        return AnalysisResponse(
                            hallucination_score=0.0,
                            confidence=1.0,
                            raw_logits=[10.0, -10.0],
                            calibrated_score=0.0,
                            explanation=f"Detected simple arithmetic. Expression `{expr}` evaluates to {expected}, response matches.",
                            evidence=None,
                            rag_used=False
                        )
                    else:
                        return AnalysisResponse(
                            hallucination_score=10.0,
                            confidence=0.01,
                            raw_logits=[-10.0, 10.0],
                            calibrated_score=10.0,
                            explanation=f"Detected simple arithmetic. Expression `{expr}` evaluates to {expected}, but response contains {resp_val}.",
                            evidence=None,
                            rag_used=False
                        )

        # ---------------- Build enhanced premise ----------------
        evidence = None
        rag_used = False
        if request.use_rag and rag_retriever is not None:
            rag_used = True
            evidence = rag_retriever.retrieve(request.response, k=3)
            if evidence:
                evidence_text = " ".join([e.get('document', '') for e in evidence])
                enhanced_premise = f"{request.prompt}\nRelevant facts: {evidence_text}"
            else:
                enhanced_premise = request.prompt
        else:
            enhanced_premise = request.prompt

        # ---------------- Compute hallucination ----------------
        result = compute_hhem_score(
            premise=enhanced_premise,
            hypothesis=request.response
        )

        # Use GPT score directly as calibrated
        calibrated_score = result["hallucination_score"]

        # ---------------- Build explanation ----------------
        if calibrated_score <= 3.0:
            explanation = "Low hallucination risk. The response appears well-grounded in the prompt."
        elif calibrated_score <= 6.0:
            explanation = "Moderate hallucination risk. Some claims may need verification."
        else:
            explanation = "High hallucination risk. Significant discrepancies detected between prompt and response."
        if rag_used and evidence:
            explanation += f"\nRetrieved {len(evidence)} supporting documents from knowledge base."

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
        print(f"Unexpected error in /analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))
