"""
Evaluation Script 1: Baseline Only (all-MiniLM-L6-v2)
No API tokens needed - fully local
"""

import json
import time
import numpy as np
from typing import List, Dict
from baseline_models import BaselineModels
from datetime import datetime

def run_baseline_test(test_cases: List[Dict]) -> Dict:
    """Run baseline evaluation"""
    print("\n" + "="*80)
    print("TESTING BASELINE: all-MiniLM-L6-v2 (Embedding Similarity)")
    print("="*80 + "\n")
    
    baselines = BaselineModels()
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Testing: {case['prompt'][:60]}...")
        
        try:
            result = baselines.test_baseline(
                "all_minilm_l6_v2",
                prompt=case['prompt'],
                response=case['response']
            )
            
            if result.get('status') == 'success':
                score = result['hallucination_score']
                confidence = result['confidence']
                
                # Determine prediction
                if score > 6.0:
                    predicted = "hallucinated"
                elif score < 3.0:
                    predicted = "accurate"
                else:
                    predicted = "hallucinated" if score > 4.5 else "accurate"
                
                correct = predicted == case['expected_label']
                
                # Show result
                result_symbol = "✓" if correct else "✗"
                print(f"    {result_symbol} Score: {score:.1f}/10, Expected: {case['expected_label']}, Got: {predicted}")
                
                results.append({
                    "prompt": case['prompt'][:100],
                    "expected": case['expected_label'],
                    "predicted": predicted,
                    "score": score,
                    "confidence": confidence,
                    "correct": correct,
                    "category": case.get('category', 'unknown'),
                    "model": "all-MiniLM-L6-v2",
                    "status": "success"
                })
            else:
                results.append({
                    "status": "error",
                    "error": result.get('error'),
                    "expected": case['expected_label']
                })
        except Exception as e:
            results.append({
                "status": "error",
                "error": str(e),
                "expected": case['expected_label']
            })
        
        time.sleep(0.1)
    
    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics"""
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
    
    return {
        "total_tests": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "errors": len(results) - len(valid)
    }


def calculate_ece(results: List[Dict], n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error"""
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
    print("BASELINE EVALUATION (all-MiniLM-L6-v2)")
    print("No API tokens needed - runs locally")
    print("="*80)
    
    # Load test cases
    try:
        with open('test_cases_flat.json', 'r') as f:
            test_cases = json.load(f)
        print(f"\n✓ Loaded {len(test_cases)} test cases")
    except FileNotFoundError:
        print("\n✗ test_cases_flat.json not found!")
        print("Run: python create_10_test_cases.py first")
        return
    
    # Run baseline test
    results = run_baseline_test(test_cases)
    
    # Calculate metrics
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
        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  TP: {cm['TP']}, TN: {cm['TN']}, FP: {cm['FP']}, FN: {cm['FN']}")
    else:
        print(f"✗ Error: {metrics['error']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "configuration": "baseline_all_minilm_l6_v2",
        "timestamp": timestamp,
        "metrics": metrics,
        "ece": float(ece),
        "results": results
    }
    
    filename = f"results_baseline_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {filename}")
    print("="*80)


if __name__ == "__main__":
    main()