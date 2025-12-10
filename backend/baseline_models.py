"""
Baseline model: Local hallucination detection using sentence embeddings
Returns hallucination score 0-10 (higher = more hallucinated)
Fully local, no API key required
"""

from typing import Dict
import numpy as np
from sentence_transformers import SentenceTransformer


class BaselineModels:
    """Local embedding-based hallucination baseline"""
    
    def __init__(self):
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Model loaded successfully")
    
    def test_baseline(self, baseline_name: str, prompt: str, response: str) -> Dict:
        """
        Test baseline using semantic similarity between prompt and response.
        Low similarity = high hallucination score
        """
        try:
            # Encode prompt and response to embeddings
            prompt_emb = self.model.encode(prompt, convert_to_numpy=True)
            response_emb = self.model.encode(response, convert_to_numpy=True)
            
            # Calculate cosine similarity
            from numpy.linalg import norm
            cosine_sim = np.dot(prompt_emb, response_emb) / (norm(prompt_emb) * norm(response_emb))
            
            # Convert to 0-10 scale (higher = more hallucinated)
            # If similarity is high (close to 1), hallucination is low (close to 0)
            # If similarity is low (close to 0), hallucination is high (close to 10)
            hallucination_score = float(10 * (1 - cosine_sim))
            confidence = float(cosine_sim)
            
            return {
                "hallucination_score": hallucination_score,
                "confidence": confidence,
                "model": "all-MiniLM-L6-v2",
                "raw_scores": {
                    "similarity": float(cosine_sim)
                },
                "status": "success"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model": "all-MiniLM-L6-v2"
            }


def test_baselines():
    """Quick test of embedding-based hallucination baseline"""
    baselines = BaselineModels()
    
    test_cases = [
        {
            "name": "Accurate Response",
            "prompt": "What is the capital of France?",
            "response": "Paris is the capital of France.",
            "expected": "Should score LOW (0-2)"
        },
        {
            "name": "Hallucinated Response",
            "prompt": "What is the capital of Australia?",
            "response": "Sydney is the capital of Australia.",
            "expected": "Should score HIGH (6-10)"
        },
        {
            "name": "Partially Incorrect",
            "prompt": "How many moons does Jupiter have?",
            "response": "Jupiter has 42 confirmed moons.",
            "expected": "Should score MEDIUM (3-6)"
        },
        {
            "name": "Completely Unrelated",
            "prompt": "What is the capital of France?",
            "response": "The sky is blue and grass is green.",
            "expected": "Should score VERY HIGH (8-10)"
        }
    ]
    
    print("\n" + "="*80)
    print("TESTING LOCAL EMBEDDING HALLUCINATION BASELINE")
    print("Model: all-MiniLM-L6-v2 (sentence-transformers)")
    print("Returns hallucination_score 0-10 (higher = more hallucinated)")
    print("="*80)
    
    for test_case in test_cases:
        print("\n" + "="*80)
        print(f"Test: {test_case['name']}")
        print("="*80)
        print(f"Prompt: {test_case['prompt']}")
        print(f"Response: {test_case['response']}")
        print(f"Expected: {test_case['expected']}")
        print()
        
        result = baselines.test_baseline("all_minilm_l6_v2", test_case['prompt'], test_case['response'])
        
        if result.get('status') == 'success':
            score = result['hallucination_score']
            conf = result['confidence']
            similarity = result['raw_scores']['similarity']
            
            # Interpret score
            if score < 3.0:
                interpretation = "✓ ACCURATE"
            elif score < 6.0:
                interpretation = "⚠ MODERATE RISK"
            else:
                interpretation = "✗ HIGH HALLUCINATION"
            
            print(f"Score: {score:.2f}/10 | Confidence: {conf:.2f} | {interpretation}")
            print(f"Cosine Similarity: {similarity:.3f}")
        else:
            print(f"✗ Error: {result.get('error')}")
    
    print("\n" + "="*80)
    print("✓ Baseline testing complete!")
    print("="*80)


if __name__ == "__main__":
    test_baselines()