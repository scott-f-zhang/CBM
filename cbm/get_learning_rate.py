#!/usr/bin/env python3
"""Learning rate finder for multiple datasets across different models.

This script performs quick learning rate tests (1-3 epochs) to find optimal
learning rates for different model backbones on various datasets.
"""
import os
import sys
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Tuple

# Ensure project root on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cbm import (
    get_cbm_joint,
)
from cbm.config.defaults import RunConfig

# Learning rate candidates from the paper (determined via grid search)
MODEL_CONFIG = {
    'bert-base-uncased': {
        'enabled': True,
        'lr_candidates': [1e-5, 3e-5, 5e-5, 1e-4, 5e-4],  # Sorted by value
    },
    'gpt2': {
        'enabled': True,
        'lr_candidates': [1e-5, 5e-4, 1e-4, 5e-3, 1e-3],  # Sorted by value
    },
    'lstm': {
        'enabled': True,
        'lr_candidates': [1e-4, 1e-3, 1e-2, 5e-2, 1e-1],  # Sorted by value
    },
    'roberta-base': {
        'enabled': True,
        'lr_candidates': [1e-5, 3e-5, 5e-5, 1e-4, 5e-4],  # Sorted by value
    },
}


def get_learning_rate_candidates(model_name: str) -> List[float]:
    """Get learning rate candidates for a specific model."""
    if model_name in MODEL_CONFIG:
        return MODEL_CONFIG[model_name]['lr_candidates']
    return [1e-6, 1e-5, 2e-5, 5e-5]  # Default fallback


def get_dataset_variant(dataset: str) -> str:
    """Get default variant for each dataset."""
    variant_map = {
        'essay': None,      # No variant needed
        'qa': None,         # No variant needed
        'cebab': 'aug',     # Use augmented variant
        'imdb': 'gen',      # Use generated variant
    }
    return variant_map.get(dataset, None)


def quick_lr_test(model_name: str, dataset: str, num_epochs: int = 3) -> Dict[float, Dict]:
    """
    Quick learning rate test for a specific model and dataset.
    
    Args:
        model_name: Model backbone name
        dataset: Dataset name (essay, qa, cebab, imdb)
        num_epochs: Number of epochs for quick test (default: 3)
    
    Returns:
        Dict mapping learning rate to results
    """
    print(f"\n=== Testing {model_name} on {dataset.upper()} dataset ===")
    
    lr_candidates = get_learning_rate_candidates(model_name)
    results = {}
    
    # Get variant for dataset
    variant = get_dataset_variant(dataset)
    
    # Base configuration
    base_config = {
        'model_name': model_name,
        'dataset': dataset,
        'num_epochs': num_epochs,
        'max_len': 512,
        'batch_size': 8,
    }
    
    # Add variant if needed
    if variant is not None:
        base_config['variant'] = variant
    
    for lr in lr_candidates:
        print(f"Testing LR: {lr:.0e}")
        
        try:
            # Use joint pipeline as standard method
            result = get_cbm_joint(optimizer_lr=lr, **base_config)
            
            # Joint pipeline returns dict with 'task' and 'concept' keys
            if isinstance(result, dict):
                task_scores = result.get('task', [])
                concept_scores = result.get('concept', [])
                
                if task_scores and len(task_scores) > 0:
                    task_acc, task_f1 = task_scores[0]
                else:
                    task_acc, task_f1 = 0.0, 0.0
                    
                if concept_scores and len(concept_scores) > 0:
                    concept_acc, concept_f1 = concept_scores[0]
                else:
                    concept_acc, concept_f1 = 0.0, 0.0
            else:
                task_acc, task_f1 = 0.0, 0.0
                concept_acc, concept_f1 = 0.0, 0.0
            
            results[lr] = {
                'task_acc': task_acc,
                'task_f1': task_f1,
                'concept_acc': concept_acc,
                'concept_f1': concept_f1,
                'combined_score': task_acc + concept_acc,  # Simple combination
            }
            
            print(f"  Task: {task_acc:.4f}/{task_f1:.4f}, "
                  f"Concept: {concept_acc:.4f}/{concept_f1:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[lr] = {
                'task_acc': 0.0,
                'task_f1': 0.0,
                'concept_acc': 0.0,
                'concept_f1': 0.0,
                'combined_score': 0.0,
            }
    
    return results


def find_best_lr(results: Dict[float, Dict], metric: str = 'combined_score') -> Tuple[float, Dict]:
    """Find the best learning rate based on specified metric."""
    if not results:
        return None, {}
    
    best_lr = max(results.keys(), key=lambda lr: results[lr][metric])
    best_result = results[best_lr]
    
    return best_lr, best_result


def run_lr_finder(dataset: str):
    """Run learning rate finder for all models on specified dataset."""
    
    # Models to test (only enabled ones)
    models = [model for model, config in MODEL_CONFIG.items() if config.get('enabled', False)]
    
    # Results storage
    all_results = {}
    
    print("=" * 60)
    print(f"LEARNING RATE FINDER FOR {dataset.upper()} DATASET")
    print("=" * 60)
    
    for model in models:
        print(f"\n{'='*20} {model.upper()} {'='*20}")
        
        # Run quick test
        results = quick_lr_test(model, dataset, num_epochs=3)
        
        # Find best learning rate
        best_lr, best_result = find_best_lr(results, 'combined_score')
        
        all_results[model] = {
            'all_results': results,
            'best_lr': best_lr,
            'best_result': best_result,
        }
        
        if best_lr:
            print(f"\nBest LR for {model}: {best_lr:.0e}")
            print(f"Best scores: Task={best_result['task_acc']:.4f}, "
                  f"Concept={best_result['concept_acc']:.4f}")
        else:
            print(f"\nNo valid results for {model}")
    
    return all_results


def save_results(results: Dict, output_file: str):
    """Save learning rate finder results to CSV (simplified format)."""
    
    rows = []
    
    for model, model_results in results.items():
        best_lr = model_results.get('best_lr')
        
        if best_lr:
            rows.append({
                'model': model,
                'best_lr': best_lr,
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df


def print_summary(results: Dict, dataset: str):
    """Print a summary of the best learning rates."""
    
    print("\n" + "=" * 60)
    print(f"LEARNING RATE SUMMARY FOR {dataset.upper()}")
    print("=" * 60)
    
    for model, model_results in results.items():
        best_lr = model_results.get('best_lr')
        best_result = model_results.get('best_result', {})
        
        if best_lr:
            print(f"{model.upper()}: LR={best_lr:.0e}, "
                  f"Task={best_result.get('task_acc', 0.0):.4f}, "
                  f"Concept={best_result.get('concept_acc', 0.0):.4f}")
        else:
            print(f"{model.upper()}: No valid results")


def main():
    """Main function to run learning rate finder."""
    
    parser = argparse.ArgumentParser(
        description="Find optimal learning rates for different datasets and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_learning_rate.py --dataset essay
  python get_learning_rate.py --dataset qa
  python get_learning_rate.py --dataset cebab
  python get_learning_rate.py --dataset imdb
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['essay', 'qa', 'cebab', 'imdb'],
        help='Dataset to find learning rates for'
    )
    
    args = parser.parse_args()
    
    # Run learning rate finder
    results = run_lr_finder(args.dataset)
    
    # Save results to lr_rate directory
    MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
    lr_rate_dir = os.path.join(MAIN_DIR, "lr_rate")
    os.makedirs(lr_rate_dir, exist_ok=True)
    output_file = os.path.join(lr_rate_dir, f"{args.dataset}_lr_rate.csv")
    
    df = save_results(results, output_file)
    
    # Print summary
    print_summary(results, args.dataset)
    
    # Print detailed results table
    print("\n" + "=" * 60)
    print("DETAILED RESULTS TABLE")
    print("=" * 60)
    print(df.to_string(index=False, float_format='%.6f'))
    
    return results, df


if __name__ == "__main__":
    results, df = main()
