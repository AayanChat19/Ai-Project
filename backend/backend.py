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

from rag_retriever import EnhancedRAGRetriever, create_enhanced_knowledge_base

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

rag_retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_retriever
    try:
        print("\n" + "="*60)
        print("🚀 Starting up Enhanced Hallucination Detector API...")
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
    
    yield
    
    print("\n" + "="*60)
    print("🛑 Shutting down Hallucination Detector API...")
    print("="*60 + "\n")

app = FastAPI(
    title="Enhanced Hallucination Detector API with Claim-Level Analysis",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CLAIM EXTRACTION AND VERIFICATION
# ============================================================================

def extract_claims(response: str) -> List[str]:
    """
    Extract individual factual claims from a response.
    Split on sentence boundaries and filter for factual statements.
    """
    try:
        prompt = f"""Extract all factual claims from this text.

Text: "{response[:500]}"

Extract claims that contain:
- Specific facts, numbers, dates, or names
- Verifiable statements
- Concrete information

Ignore opinions, questions, and greetings.

CRITICAL: Return ONLY a valid JSON array of strings. No markdown, no explanations.
Format: ["claim 1", "claim 2", "claim 3"]"""

        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        text = completion.choices[0].message.content.strip()
        
        # Remove markdown
        text = re.sub(r'```json\s*|\s*```', '', text)
        text = text.strip()
        
        # Try direct parse first
        try:
            result = json.loads(text)
            # Handle both {"claims": [...]} and direct [...]
            if isinstance(result, dict):
                claims = result.get('claims', result.get('items', []))
            elif isinstance(result, list):
                claims = result
            else:
                raise ValueError("Invalid JSON structure")
        except json.JSONDecodeError:
            # Fallback: extract array with regex
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                claims = json.loads(match.group())
            else:
                raise ValueError("No JSON array found")
        
        if isinstance(claims, list) and len(claims) > 0:
            # Filter out empty strings
            claims = [c.strip() for c in claims if isinstance(c, str) and len(c.strip()) > 5]
            print(f"✓ Extracted {len(claims)} claims from response")
            return claims
        
        # If no valid claims, return full response
        print("⚠ No claims extracted, using full response")
        return [response]
        
    except Exception as e:
        print(f"⚠ Claim extraction error: {e}, using sentence splitting")
        # Fallback: split by sentence
        sentences = re.split(r'[.!?]+', response)
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims[:5] if claims else [response]  # Limit to 5 claims max in fallback


def verify_claim_with_evidence(claim: str, evidence_list: List[Dict]) -> Dict:
    """
    Check if a specific claim is supported by the evidence using embeddings.
    Returns support score and matching evidence.
    """
    if not evidence_list or not rag_retriever:
        return {
            "claim": claim,
            "supported": False,
            "confidence": 0.0,
            "matching_evidence": []
        }
    
    try:
        # Encode claim
        claim_embedding = rag_retriever.embedder.encode([claim])
        
        # Check against each evidence
        matches = []
        for ev in evidence_list:
            ev_text = ev.get('document', '')
            if not ev_text:
                continue
                
            ev_embedding = rag_retriever.embedder.encode([ev_text])
            
            # Cosine similarity
            from numpy.linalg import norm
            similarity = float(
                (claim_embedding @ ev_embedding.T) / 
                (norm(claim_embedding) * norm(ev_embedding))
            )
            
            if similarity > 0.5:  # Threshold for relevance
                matches.append({
                    "evidence": ev_text[:200],
                    "similarity": similarity,
                    "source": ev.get('metadata', {}).get('source', 'Unknown'),
                    "source_type": ev.get('source_type', 'unknown')
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        supported = len(matches) > 0 and matches[0]['similarity'] > 0.65
        confidence = matches[0]['similarity'] if matches else 0.0
        
        return {
            "claim": claim,
            "supported": supported,
            "confidence": float(confidence),
            "matching_evidence": matches[:2]  # Top 2 matches
        }
        
    except Exception as e:
        print(f"Claim verification error: {e}")
        return {
            "claim": claim,
            "supported": False,
            "confidence": 0.0,
            "matching_evidence": []
        }


def deep_search_for_claim(claim: str) -> List[Dict]:
    """
    Perform targeted web search for a specific unsupported claim.
    """
    if not rag_retriever or not rag_retriever.enable_web_search:
        return []
    
    print(f"\n🔍 Deep search for unsupported claim: {claim[:80]}...")
    
    # Search using the claim itself
    results = rag_retriever.retrieve_web(claim, max_results=3)
    
    if results:
        print(f"✓ Found {len(results)} additional sources for claim")
    else:
        print("⚠ No additional sources found")
    
    return results


# ============================================================================
# ENHANCED EVIDENCE RETRIEVAL
# ============================================================================

def retrieve_evidence_multi_query(prompt: str, response: str, use_rag: bool, 
                                   use_web_search: bool) -> tuple:
    """
    Retrieve evidence using BOTH prompt and response as queries.
    Returns: (evidence_list, evidence_text, rag_used, web_search_used)
    """
    evidence = []
    rag_used = False
    web_search_used = False
    
    if not rag_retriever or not use_rag:
        return evidence, "", rag_used, web_search_used
    
    rag_used = True
    original_web_setting = rag_retriever.enable_web_search
    
    if not use_web_search:
        rag_retriever.enable_web_search = False
    
    try:
        # Search 1: Based on the PROMPT (what was asked)
        print(f"\n{'='*60}")
        print("📚 Searching based on PROMPT...")
        prompt_evidence = rag_retriever.retrieve(prompt, k=3)
        
        # Search 2: Based on the RESPONSE (what was claimed)
        print(f"\n{'='*60}")
        print("📝 Searching based on RESPONSE...")
        response_evidence = rag_retriever.retrieve(response, k=3)
        
        # Combine and deduplicate
        seen_docs = set()
        for ev in prompt_evidence + response_evidence:
            doc_text = ev.get('document', '')
            if doc_text not in seen_docs:
                evidence.append(ev)
                seen_docs.add(doc_text)
        
        # Check if web search was used
        if evidence:
            web_search_used = any(e.get('source_type') == 'web' for e in evidence)
        
        # Sort by relevance and source reliability
        source_priority = {'local': 3, 'web': 2, 'unknown': 1}
        evidence.sort(
            key=lambda x: (source_priority.get(x.get('source_type', 'unknown'), 0), 
                          x.get('relevance', 0)),
            reverse=True
        )
        
        evidence = evidence[:5]  # Top 5 overall
        
        evidence_text = "\n".join([
            f"{i+1}. [{e.get('source_type', 'unknown').upper()}] "
            f"{e.get('document', '')} "
            f"(Relevance: {e.get('relevance', 0):.2f}, "
            f"Source: {e.get('metadata', {}).get('source', 'Unknown')})"
            for i, e in enumerate(evidence)
        ])
        
        local_count = sum(1 for e in evidence if e.get('source_type') == 'local')
        web_count = sum(1 for e in evidence if e.get('source_type') == 'web')
        
        print(f"\n✓ Retrieved {len(evidence)} total documents (Local: {local_count}, Web: {web_count})")
        
    except Exception as e:
        print(f"Evidence retrieval error: {e}")
    finally:
        rag_retriever.enable_web_search = original_web_setting
    
    return evidence, evidence_text, rag_used, web_search_used


# ============================================================================
# IMPROVED SCORING WITH CLAIM-LEVEL ANALYSIS
# ============================================================================

def compute_openai_score(premise: str, hypothesis: str, evidence_text: str = "",
                        claim_analysis: Optional[Dict] = None):
    """
    Enhanced OpenAI scoring with claim-level analysis results.
    Uses structured output for reliable JSON.
    """
    evidence_section = ""
    if evidence_text:
        evidence_section = f"\n\nRelevant Evidence:\n{evidence_text[:800]}\n"
    
    claim_section = ""
    if claim_analysis:
        unsupported = claim_analysis.get('unsupported_claims', [])
        if unsupported:
            claim_section = f"\n\n⚠️ Unsupported Claims Detected:\n"
            for i, claim_info in enumerate(unsupported[:3], 1):
                claim_section += f"{i}. {claim_info['claim'][:100]}\n"
    
    prompt = f"""You are a hallucination detector. Analyze if the Response contains false, unsupported, or made-up information.

Prompt: "{premise[:400]}"
Response: "{hypothesis[:400]}"{evidence_section[:1000]}{claim_section}

Analyze if the Response contains hallucinations. Use the evidence to verify claims.

CRITICAL: Return ONLY valid JSON with these exact keys:
- hallucination_score: number 0-10
- confidence: number 0.0-1.0  
- reasoning: brief explanation (1-2 sentences)

Scoring:
- 0-2: Accurate and well-supported
- 3-5: Minor issues or unsupported claims
- 6-8: Significant false information
- 9-10: Mostly fabricated

Example: {{"hallucination_score": 3.5, "confidence": 0.85, "reasoning": "Most claims are accurate but one statement lacks support."}}"""

    try:
        print(f"\n{'='*60}")
        print(f"Calling OpenAI API...")
        
        if not openai.api_key:
            raise ValueError("OpenAI API key not set")
        
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"}
        )

        text = completion.choices[0].message.content.strip()
        print(f"OpenAI raw output: {text[:200]}")

        # Remove markdown if present
        text = re.sub(r'```json\s*|\s*```', '', text)
        text = text.strip()
        
        # Try direct parse
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: extract JSON with regex
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
            if match:
                result = json.loads(match.group())
            else:
                raise ValueError("No valid JSON found")
        
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

    except Exception as e:
        print(f"✗ OpenAI API Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


def gemini_judge(premise: str, hypothesis: str, openai_result: Dict, 
                evidence: List[Dict], claim_analysis: Optional[Dict] = None) -> Dict:
    """
    Enhanced Gemini judge with claim-level analysis.
    Uses gemini-2.5-flash with improved JSON extraction.
    """
    evidence_text = "\n".join([
        f"- [{e.get('source_type', 'unknown').upper()}] {e.get('document', '')[:150]}" 
        for e in evidence
    ]) if evidence else "No evidence"
    
    claim_section = ""
    if claim_analysis:
        total = claim_analysis.get('total_claims', 0)
        supported = claim_analysis.get('supported_count', 0)
        unsupported = claim_analysis.get('unsupported_count', 0)
        
        claim_section = f"""

Claim Analysis:
- Total claims: {total}
- Supported: {supported}
- Unsupported: {unsupported}
- Support rate: {(supported/total*100) if total > 0 else 0:.1f}%
"""
    
    prompt = f"""You are an LLM Judge evaluating hallucination detection.

CRITICAL: You MUST respond with ONLY a valid JSON object. No markdown, no code blocks, no explanations.

SCALE: 0 = fully ACCURATE, 10 = completely FABRICATED

Required JSON format:
{{
    "final_score": <number 0-10>, 
    "agreement": "agree/disagree", 
    "judge_reasoning": "Brief explanation here", 
    "confidence": <number 0.0-1.0>
}}

Prompt: "{premise[:300]}"
Response: "{hypothesis[:300]}"

Evidence:
{evidence_text[:500]}{claim_section}

OpenAI Analysis:
- Score: {openai_result['hallucination_score']}/10
- Confidence: {openai_result['confidence']}
- Reasoning: {openai_result.get('reasoning', 'N/A')[:150]}

Task: Review and either agree with or adjust OpenAI's score. Return ONLY the JSON object."""

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


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ClaimVerification(BaseModel):
    claim: str
    supported: bool
    confidence: float
    matching_evidence: List[Dict]

class AnalysisRequest(BaseModel):
    prompt: str
    response: str
    use_rag: bool = True
    use_web_search: bool = True
    use_claim_analysis: bool = True
    context: Optional[str] = None

class AnalysisResponse(BaseModel):
    hallucination_score: float
    confidence: float
    raw_logits: List[float]
    calibrated_score: Optional[float] = None
    evidence: Optional[List[Dict]] = None
    explanation: str
    rag_used: bool = False
    web_search_used: bool = False
    claim_analysis: Optional[Dict] = None
    judge_explanation: Optional[str] = None
    openai_score: Optional[float] = None
    gemini_score: Optional[float] = None
    analysis_style: Optional[str] = None


# ============================================================================
# MAIN ENDPOINT
# ============================================================================

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_hallucination(request: AnalysisRequest):
    try:
        print(f"\n{'='*80}")
        print(f"NEW ANALYSIS REQUEST")
        print(f"{'='*80}")
        print(f"Prompt: {request.prompt[:150]}...")
        print(f"Response: {request.response[:150]}...")
        print(f"Claim Analysis: {request.use_claim_analysis}")
        print(f"{'='*80}")

        # Retrieve evidence using both prompt and response
        evidence, evidence_text, rag_used, web_search_used = retrieve_evidence_multi_query(
            request.prompt, request.response, request.use_rag, request.use_web_search
        )

        # Claim-level analysis
        claim_analysis_results = None
        if request.use_claim_analysis:
            print(f"\n{'='*60}")
            print("🔬 Performing Claim-Level Analysis...")
            print('='*60)
            
            claims = extract_claims(request.response)
            
            claim_verifications = []
            unsupported_claims = []
            
            for claim in claims:
                verification = verify_claim_with_evidence(claim, evidence)
                claim_verifications.append(verification)
                
                if not verification['supported']:
                    unsupported_claims.append(verification)
                    
                    # Deep search for unsupported claims
                    additional_evidence = deep_search_for_claim(claim)
                    if additional_evidence:
                        # Re-verify with new evidence
                        verification_retry = verify_claim_with_evidence(
                            claim, evidence + additional_evidence
                        )
                        if verification_retry['supported']:
                            # Update evidence list
                            evidence.extend(additional_evidence)
                            web_search_used = True
            
            supported_count = sum(1 for v in claim_verifications if v['supported'])
            
            claim_analysis_results = {
                "total_claims": len(claims),
                "supported_count": supported_count,
                "unsupported_count": len(unsupported_claims),
                "support_rate": supported_count / len(claims) if claims else 1.0,
                "claims": claim_verifications,
                "unsupported_claims": unsupported_claims
            }
            
            print(f"\n📊 Claim Analysis Summary:")
            print(f"   Total: {len(claims)} | Supported: {supported_count} | Unsupported: {len(unsupported_claims)}")

        # OpenAI Analysis
        openai_result = compute_openai_score(
            premise=request.prompt,
            hypothesis=request.response,
            evidence_text=evidence_text,
            claim_analysis=claim_analysis_results
        )

        # Gemini Judge
        judge_result = gemini_judge(
            premise=request.prompt,
            hypothesis=request.response,
            openai_result=openai_result,
            evidence=evidence or [],
            claim_analysis=claim_analysis_results
        )

        # Build explanation
        final_score = judge_result['final_score']
        
        if final_score <= 3.0:
            base_explanation = "✓ Low hallucination risk. Response appears accurate."
        elif final_score <= 6.0:
            base_explanation = "⚠ Moderate hallucination risk. Some claims need verification."
        else:
            base_explanation = "✗ High hallucination risk. Significant issues detected."
        
        if judge_result['agreement'] == 'adjusted':
            judge_note = f"\n\n🔍 LLM Judge: Adjusted from {openai_result['hallucination_score']:.1f}/10 to {final_score:.1f}/10"
        else:
            judge_note = f"\n\n🔍 LLM Judge: Confirmed assessment"
        
        explanation = base_explanation + judge_note + "\n\n" + openai_result.get('reasoning', '')

        if claim_analysis_results:
            support_rate = claim_analysis_results['support_rate'] * 100
            explanation += f"\n\n🔬 Claim Analysis: {support_rate:.0f}% of claims supported by evidence"

        if rag_used and evidence:
            local_count = sum(1 for e in evidence if e.get('source_type') == 'local')
            web_count = sum(1 for e in evidence if e.get('source_type') == 'web')
            explanation += f"\n\n📚 Evidence: {local_count} local + 🌐 {web_count} web sources"

        score_diff = abs(openai_result['hallucination_score'] - final_score)
        analysis_style = "Deterministic" if score_diff <= 1.0 else "Balanced" if score_diff <= 3.0 else "Creative"

        print(f"\n{'='*80}")
        print(f"FINAL RESULTS:")
        print(f"Score: {final_score}/10 | Style: {analysis_style} | Web: {web_search_used}")
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
            claim_analysis=claim_analysis_results,
            judge_explanation=judge_result['judge_reasoning'],
            openai_score=openai_result['hallucination_score'],
            gemini_score=final_score,
            analysis_style=analysis_style
        )

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_initialized": rag_retriever is not None,
        "web_search_enabled": rag_retriever.enable_web_search if rag_retriever else False,
        "claim_analysis_available": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)