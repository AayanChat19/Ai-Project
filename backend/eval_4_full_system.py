"""
Evaluation Script 4: Your System - Full (RAG + Web Search + Claim Analysis)
Tests: Complete system with all features
"""

import json
import requests
import time
import numpy as np
from typing import List, Dict
from datetime import datetime

API_URL = "http://localhost:8000"

def test_full_system(test_cases: List[Dict]) -> List[Dict]:
    """Test full system with all features"""
    print("\n" + "="*80)
    print("TESTING YOUR SYSTEM: FULL (RAG + Web Search + Claim Analysis)")
    print("="*80 + "\n")
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Testing: {case['prompt'][:60]}...")
        
        try:
            payload = {
                "prompt": case['prompt'],
                "response": case['response'],
                "use_rag": True,
                "use_web_search": True,
                "use_claim_analysis": True
            }
            
            response = requests.post(f"{API_URL}/analyze", json=payload, timeout=90)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'hallucination_score' not in data:
                    print(f"    ✗ Missing hallucination_score")
                    results.append({"status": "error", "error": "Missing score", "expected": case['expected_label']})
                    continue
                
                score = data['hallucination_score']
                confidence = data.get('confidence', 0.5)
                rag_used = data.get('rag_used', False)
                web_used = data.get('web_search_used', False)
                evidence_count = len(data.get('evidence', []) or [])
                
                if score > 6.0:
                    predicted = "hallucinated"
                elif score < 3.0:
                    predicted = "accurate"
                else:
                    predicted = "hallucinated" if score > 4.5 else "accurate"
                
                correct = predicted == case['expected_label']
                result_symbol = "✓" if correct else "✗"
                web_symbol = "🌐" if web_used else "📚"
                print(f"    {result_symbol} Score: {score:.1f}/10, {web_symbol} Evidence: {evidence_count}, Expected: {case['expected_label']}, Got: {predicted}")
                
                results.append({
                    "prompt": case['prompt'][:100],
                    "expected": case['expected_label'],
                    "predicted": predicted,
                    "score": score,
                    "confidence": confidence,
                    "correct": correct,
                    "rag_used": rag_used,
                    "web_search_used": web_used,
                    "evidence_count": evidence_count,
                    "category": case.get('category', 'unknown'),
                    "status": "success"
                })
            else:
                print(f"    ✗ API error: {response.status_code}")
                results.append({"status": "error", "error": f"Status {response.status_code}", "expected": case['expected_label']})
        
        except requests.Timeout:
            print(f"    ✗ Timeout (>90s)")
            results.append({"status": "error", "error": "Timeout", "expected": case['expected_label']})
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results.append({"status": "error", "error": str(e), "expected": case['expected_label']})
        
        time.sleep(1)
    
    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate metrics"""
    valid = [r for r in results if r.get('status') == 'success']
    
    if not valid:
        return {"error": "No valid results"}
    
    correct = sum(1 for r in valid if r['correct'])
    total = len(valid)
    accuracy = correct / total
    
    tp = sum(1 for r in valid if r['expected'] == 'hallucinated' and r['predicted'] == 'hallucinated')
    tn = sum(1 for r in valid if r['expected'] == 'accurate' and r['predicted'] == 'accurate')
    fp = sum(1 for r in valid if r['expected'] == 'accurate' and r['predicted'] == 'hallucinated')
    fn = sum(1 for r in valid if r['expected'] == 'hallucinated' and r['predicted'] == 'accurate')
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Feature usage stats
    rag_used_count = sum(1 for r in valid if r.get('rag_used', False))
    web_used_count = sum(1 for r in valid if r.get('web_search_used', False))
    avg_evidence = np.mean([r.get('evidence_count', 0) for r in valid])
    
    return {
        "total_tests": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "rag_usage": rag_used_count / total if total > 0 else 0,
        "web_search_usage": web_used_count / total if total > 0 else 0,
        "avg_evidence": float(avg_evidence),
        "errors": len(results) - len(valid)
    }


def calculate_ece(results: List[Dict], n_bins: int = 10) -> float:
    """Calculate ECE"""
    valid = [r for r in results if r.get('status') == 'success' and 'confidence' in r]
    
    if not valid:
        return 0.0
    
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_results = [r for r in valid if bins[i] <= r['confidence'] < bins[i+1]]
        if bin_results:
            bin_accuracy = sum(1 for r in bin_results if r['correct']) / len(bin_results)
            bin_confidence = np.mean([r['confidence'] for r in bin_results])
            ece += abs(bin_accuracy - bin_confidence) * (len(bin_results) / len(valid))
    
    return ece


def main():
    print("\n" + "="*80)
    print("YOUR SYSTEM EVALUATION - FULL SYSTEM")
    print("Uses OpenAI + Gemini tokens + RAG + Web Search")
    print("="*80)
    
    # Check backend
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code != 200:
            print("\n✗ Backend not healthy")
            return
        print("\n✓ Backend is healthy")
    except:
        print("\n✗ Cannot connect to backend")
        print("Start backend: python backend.py")
        return
    
    # Load test cases
    try:
        with open('test_cases_flat.json', 'r') as f:
            test_cases = json.load(f)
        print(f"✓ Loaded {len(test_cases)} test cases")
    except FileNotFoundError:
        print("\n✗ test_cases_flat.json not found!")
        return
    
    # Run test
    results = test_full_system(test_cases)
    metrics = calculate_metrics(results)
    ece = calculate_ece(results)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    
    if "error" not in metrics:
        print(f"Accuracy:  {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1 Score:  {metrics['f1_score']:.3f}")
        print(f"ECE:       {ece:.3f}")
        print(f"\nFeature Usage:")
        print(f"  RAG Usage: {metrics['rag_usage']:.1%}")
        print(f"  Web Search Usage: {metrics['web_search_usage']:.1%}")
        print(f"  Avg Evidence: {metrics['avg_evidence']:.1f}")
        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  TP: {cm['TP']}, TN: {cm['TN']}, FP: {cm['FP']}, FN: {cm['FN']}")
    else:
        print(f"✗ Error: {metrics['error']}")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "configuration": "your_system_full",
        "timestamp": timestamp,
        "metrics": metrics,
        "ece": float(ece),
        "results": results
    }
    
    filename = f"results_full_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {filename}")
    print("="*80)


if __name__ == "__main__":
    main()
    