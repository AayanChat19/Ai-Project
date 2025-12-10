"""
Enhanced Evaluation Script: Compare All Results with Visualizations
Creates comparison tables, charts, and exports publication-ready figures
"""

import json
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_latest_results():
    """Load the most recent result file for each configuration"""
    
    configs = {
        "baseline": "results_baseline_*.json",
        "no_rag": "results_no_rag_*.json",
        "rag_local": "results_rag_local_*.json",
        "full": "results_full_*.json"
    }
    
    results = {}
    
    for name, pattern in configs.items():
        files = glob.glob(pattern)
        if files:
            # Get most recent file
            latest = max(files, key=lambda x: x)
            
            with open(latest, 'r') as f:
                data = json.load(f)
            
            results[name] = {
                "file": latest,
                "metrics": data.get("metrics", {}),
                "ece": data.get("ece", 0.0),
                "timestamp": data.get("timestamp", "")
            }
            print(f"✓ Loaded {name}: {latest}")
        else:
            print(f"⚠ No results found for {name}")
    
    return results


def create_metrics_comparison_chart(results):
    """Create bar chart comparing key metrics"""
    
    config_names = {
        "baseline": "Baseline\n(MiniLM-L6)",
        "no_rag": "No RAG",
        "rag_local": "RAG Local",
        "full": "Full System"
    }
    
    # Prepare data
    configs = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for key in ["baseline", "no_rag", "rag_local", "full"]:
        if key in results:
            configs.append(config_names[key])
            m = results[key]["metrics"]
            accuracies.append(m.get("accuracy", 0))
            precisions.append(m.get("precision", 0))
            recalls.append(m.get("recall", 0))
            f1_scores.append(m.get("f1_score", 0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(configs))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#3b82f6')
    bars2 = ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#10b981')
    bars3 = ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#f59e0b')
    bars4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1 Score', color='#ef4444')
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Hallucination Detection Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: metrics_comparison.png")
    plt.close()


def create_ece_comparison_chart(results):
    """Create bar chart for ECE (Expected Calibration Error)"""
    
    config_names = {
        "baseline": "Baseline\n(MiniLM-L6)",
        "no_rag": "No RAG",
        "rag_local": "RAG Local",
        "full": "Full System"
    }
    
    configs = []
    eces = []
    
    for key in ["baseline", "no_rag", "rag_local", "full"]:
        if key in results:
            configs.append(config_names[key])
            eces.append(results[key]["ece"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#ef4444' if ece > 0.2 else '#f59e0b' if ece > 0.1 else '#10b981' for ece in eces]
    bars = ax.bar(configs, eces, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Calibration Comparison (Lower is Better)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, ece in zip(bars, eces):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{ece:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add reference line for "good" calibration
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good Calibration (< 0.1)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('ece_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: ece_comparison.png")
    plt.close()


def create_confusion_matrix_heatmaps(results):
    """Create confusion matrix heatmaps for all configurations"""
    
    config_names = {
        "baseline": "Baseline (MiniLM-L6)",
        "no_rag": "No RAG",
        "rag_local": "RAG Local",
        "full": "Full System"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, key in enumerate(["baseline", "no_rag", "rag_local", "full"]):
        if key in results:
            cm = results[key]["metrics"].get("confusion_matrix", {})
            
            # Create confusion matrix array
            matrix = np.array([
                [cm.get("TN", 0), cm.get("FP", 0)],
                [cm.get("FN", 0), cm.get("TP", 0)]
            ])
            
            # Create heatmap
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=True, 
                       xticklabels=['Predicted No Hallucination', 'Predicted Hallucination'],
                       yticklabels=['Actual No Hallucination', 'Actual Hallucination'],
                       annot_kws={'fontsize': 14, 'fontweight': 'bold'})
            
            axes[idx].set_title(config_names[key], fontsize=12, fontweight='bold', pad=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
            axes[idx].set_ylabel('True Label', fontsize=10)
    
    plt.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrices.png")
    plt.close()


def create_feature_usage_chart(results):
    """Create chart showing RAG and Web Search usage"""
    
    configs = []
    rag_usage = []
    web_usage = []
    
    config_names = {
        "no_rag": "No RAG",
        "rag_local": "RAG Local",
        "full": "Full System"
    }
    
    for key in ["no_rag", "rag_local", "full"]:
        if key in results:
            m = results[key]["metrics"]
            configs.append(config_names[key])
            rag_usage.append(m.get("rag_usage", 0) * 100)
            web_usage.append(m.get("web_search_usage", 0) * 100)
    
    if not configs:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rag_usage, width, label='RAG Usage', color='#8b5cf6')
    bars2 = ax.bar(x + width/2, web_usage, width, label='Web Search Usage', color='#06b6d4')
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Usage Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Usage Across Configurations', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{height:.0f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feature_usage.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_usage.png")
    plt.close()


def create_summary_table_image(results):
    """Create a publication-ready summary table as an image"""
    
    config_names = {
        "baseline": "Baseline (MiniLM-L6)",
        "no_rag": "No RAG",
        "rag_local": "RAG Local",
        "full": "Full System"
    }
    
    # Prepare data
    table_data = []
    
    for key in ["baseline", "no_rag", "rag_local", "full"]:
        if key in results:
            m = results[key]["metrics"]
            ece = results[key]["ece"]
            
            row = [
                config_names[key],
                f"{m.get('accuracy', 0):.3f}",
                f"{m.get('precision', 0):.3f}",
                f"{m.get('recall', 0):.3f}",
                f"{m.get('f1_score', 0):.3f}",
                f"{ece:.3f}"
            ]
            table_data.append(row)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    columns = ['Configuration', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ECE']
    
    table = ax.table(cellText=table_data, colLabels=columns, 
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4f46e5')
        cell.set_text_props(weight='bold', color='white')
    
    # Color rows
    colors = ['#f3f4f6', 'white']
    for i, row in enumerate(table_data):
        for j in range(len(columns)):
            cell = table[(i+1, j)]
            cell.set_facecolor(colors[i % 2])
            
            # Highlight best scores in green
            if j > 0:  # Skip configuration name
                try:
                    value = float(row[j])
                    # For ECE, lower is better
                    if j == 5:  # ECE column
                        if value < 0.1:
                            cell.set_facecolor('#d1fae5')
                    else:  # Other metrics, higher is better
                        if value >= 0.95:
                            cell.set_facecolor('#d1fae5')
                except:
                    pass
    
    plt.title('Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('summary_table.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: summary_table.png")
    plt.close()


def create_radar_chart(results):
    """Create radar chart comparing configurations"""
    
    config_names = {
        "baseline": "Baseline",
        "no_rag": "No RAG",
        "rag_local": "RAG Local",
        "full": "Full System"
    }
    
    # Metrics to compare
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['#ef4444', '#3b82f6', '#8b5cf6', '#10b981']
    
    for idx, (key, color) in enumerate(zip(["baseline", "no_rag", "rag_local", "full"], colors)):
        if key in results:
            m = results[key]["metrics"]
            values = [
                m.get('accuracy', 0),
                m.get('precision', 0),
                m.get('recall', 0),
                m.get('f1_score', 0)
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=config_names[key], color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax.set_title('Multi-Metric Performance Comparison', fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: radar_chart.png")
    plt.close()


def print_comparison_table(results):
    """Print formatted comparison table to console"""
    
    print("\n" + "="*100)
    print("FINAL COMPARISON TABLE")
    print("="*100)
    
    print(f"\n{'Configuration':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ECE':<10}")
    print("-" * 100)
    
    if "baseline" in results:
        m = results["baseline"]["metrics"]
        ece = results["baseline"]["ece"]
        print(f"{'Baseline (MiniLM-L6)':<25} {m.get('accuracy', 0):<12.3f} {m.get('precision', 0):<12.3f} "
              f"{m.get('recall', 0):<12.3f} {m.get('f1_score', 0):<12.3f} {ece:<10.3f}")
    
    print("-" * 100)
    
    system_configs = [
        ("no_rag", "Your System: No RAG"),
        ("rag_local", "Your System: RAG Local"),
        ("full", "Your System: Full")
    ]
    
    for key, label in system_configs:
        if key in results:
            m = results[key]["metrics"]
            ece = results[key]["ece"]
            print(f"{label:<25} {m.get('accuracy', 0):<12.3f} {m.get('precision', 0):<12.3f} "
                  f"{m.get('recall', 0):<12.3f} {m.get('f1_score', 0):<12.3f} {ece:<10.3f}")
    
    print("="*100)


def save_summary(results):
    """Save comparison summary to JSON"""
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "configurations": {}
    }
    
    for name, data in results.items():
        summary["configurations"][name] = {
            "metrics": data["metrics"],
            "ece": data["ece"],
            "source_file": data["file"]
        }
    
    filename = "comparison_summary.json"
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Comparison summary saved to {filename}")


def main():
    print("\n" + "="*100)
    print("CREATING PUBLICATION-READY VISUALIZATIONS")
    print("="*100 + "\n")
    
    # Load all results
    results = load_latest_results()
    
    if not results:
        print("\n✗ No result files found!")
        print("\nRun evaluations first:")
        print("  1. python eval_1_baseline.py")
        print("  2. python eval_2_no_rag.py")
        print("  3. python eval_3_rag_local.py")
        print("  4. python eval_4_full_system.py")
        return
    
    print("\n📊 Generating visualizations...")
    print("-" * 100)
    
    # Create all visualizations
    create_summary_table_image(results)
    create_metrics_comparison_chart(results)
    create_ece_comparison_chart(results)
    create_confusion_matrix_heatmaps(results)
    create_feature_usage_chart(results)
    create_radar_chart(results)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Save summary
    save_summary(results)
    
    print("\n" + "="*100)
    print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
    print("="*100)
    print("\nGenerated files:")
    print("  1. summary_table.png - Clean table for reports")
    print("  2. metrics_comparison.png - Bar chart of all metrics")
    print("  3. ece_comparison.png - Calibration quality comparison")
    print("  4. confusion_matrices.png - All confusion matrices")
    print("  5. feature_usage.png - RAG and web search usage")
    print("  6. radar_chart.png - Multi-dimensional comparison")
    print("  7. comparison_summary.json - Raw data export")
    print("\n💡 Tip: Use these images directly in your research paper or presentation!")


if __name__ == "__main__":
    main()